import logging
logger = logging.getLogger(__name__)

import os
import copy
from typing import List, Dict, Literal, Union, Tuple, Any, Optional
import json
import math
from dataclasses import dataclass
import pprint

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from multiprocessing import Manager
import cv2

import tqdm
import matplotlib.pyplot as plt
import matplotlib

import trimesh.transformations as tra
import trimesh
import omegaconf

import batch_urdf
import robohang.common.utils as utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.insert.net_e2e import CNNBackBone, ResnetBackBone, MLP, PositionalEncoding
from robohang.policy.insert.insert_gym_e2e import ROBOT_DIM # Action dimension
from robohang.policy.insert.insert_policy_e2e import InsertPolicyE2E, InsertGymE2E

import lightning.pytorch as pl
        

@dataclass
class TrajectoryInfo:
    path: str
    action_num: int


MAX_DEPTH = 2.0

class InsertDataset(Dataset):
    def __init__(
        self,
        data_list: List[TrajectoryInfo], 
        ds_cfg: omegaconf.DictConfig,
        name="none",
    ) -> None:
        super().__init__()

        self._data_list = Manager().list(data_list) # use Manager() to avoid memory leak issue
        self._ds_cfg = copy.deepcopy(ds_cfg)

        # calculate accumulated amount of data
        traj_len = [t.action_num for t in data_list]
        self._acc_len = np.cumsum(traj_len)
        self._size = self._acc_len[-1]

        # data cfg
        self._obs_horizon = int(ds_cfg.obs_horizon)
        self._actpred_len = int(ds_cfg.actpred_len)
        self._hxw = (int(self._ds_cfg.height), int(self._ds_cfg.width))

        # misc
        self._dtype = getattr(np, self._ds_cfg.dtype)
        self._name = name

        print(f"insert dataset {self._name}: size {self._size}")
    
    def __len__(self):
        return self._size
        
    def __getitem__(self, index: int):
        traj_idx = np.searchsorted(self._acc_len, index, side="right")
        step_idx = index - (0 if traj_idx == 0 else self._acc_len[traj_idx - 1])
        traj_info = self._data_list[traj_idx]

        # observation & state
        obs = []
        state = []
        for obs_idx in range(step_idx - self._obs_horizon + 1, step_idx + 1):
            obs_idx = max(obs_idx, 0) # clip minus number to 0

            depth = np.clip(
                cv2.resize(np.load(
                    os.path.join(traj_info.path, "obs", "depth", f"{obs_idx}.npy")
                ).astype(self._dtype), (self._hxw[1], self._hxw[0])) + 
                np.random.randn(*(self._hxw)) * self._ds_cfg.depth_noise_std,
                a_min=None, a_max=MAX_DEPTH,
            ) / MAX_DEPTH # clip depth, add noise
            mask_garment = cv2.resize(np.load(
                os.path.join(traj_info.path, "obs", "mask_garment", f"{obs_idx}.npy")
            ).astype(self._dtype), (self._hxw[1], self._hxw[0]))
            mask_hanger = cv2.resize(np.load(
                os.path.join(traj_info.path, "obs", "mask_hanger", f"{obs_idx}.npy")
            ).astype(self._dtype), (self._hxw[1], self._hxw[0]))
            obs.append(np.array([depth, mask_garment, mask_hanger]))

            # state
            state.append(np.load(os.path.join(traj_info.path, "state", f"{obs_idx}.npy")).astype(self._dtype))

        obs = np.array(obs, dtype=self._dtype) # [L, 3, H, W]
        state = np.array(state, dtype=self._dtype) # [L, R]

        # action
        action = []
        for action_idx in range(step_idx, step_idx + self._actpred_len):
            path = os.path.join(traj_info.path, "action", f"{action_idx}.npy")
            if os.path.exists(path):
                a = np.load(path).astype(self._dtype)
            else:
                a = np.zeros_like(action[-1])
            action.append(a)
        
        action = np.array(action, dtype=self._dtype) # [P, R]

        return dict(action=action, state=state, obs=obs)


def make_insert_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    ds_cfg: omegaconf.DictConfig,
):
    pattern = "^e2e_info.json$"
    def is_exclude_path(path: str):
        ans = (
            (not path.endswith("e2e_info.json")) and
            ("e2e_info.json" in os.listdir(os.path.dirname(path)))
        )
        return ans
    df = learn_utils.DataForest(data_path, [pattern], is_exclude_path=is_exclude_path)
    node_n_raw = df.get_forest_size(pattern)
    
    data_list = []
    print("scanning all trajectories ...")
    for idx in tqdm.tqdm(range(node_n_raw)):
        info_path = df.get_item(pattern, idx).file_path # xxx/yyy/e2e_info.json

        with open(info_path, "r") as f_obj:
            info = json.load(f_obj)
        step_num = info["step_cnt"]

        for batch_idx in range(info["batch_size"]):
            if not (
                info["score"]["left"][batch_idx] > 0.95 and 
                info["score"]["right"][batch_idx] > 0.95 and 
                not info["sim_error"]["all"][batch_idx]
            ):
                continue # skip fail trajectory

            data_list.append(TrajectoryInfo(
                path=os.path.join(os.path.dirname(info_path), f"{batch_idx}"), 
                action_num=step_num-1,
            ))

    total_size = len(data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    permutated = np.random.permutation(data_list).tolist() # randomly divide valid dataset and train dataset
    trds = InsertDataset(permutated[valid_size:], ds_cfg, name=f"train")
    vlds = InsertDataset(permutated[:valid_size], ds_cfg, name=f"valid")
    return trds, vlds


class InsertACTModule(pl.LightningModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        d_model = int(model_kwargs["net"]["token_dim"])
        obs_enc_name = model_kwargs["net"]["obs_enc"]["name"]
        if obs_enc_name == "cnn":
            self.obs_enc = CNNBackBone(
                input_height=model_kwargs["height"], input_width=model_kwargs["width"], 
                input_channel=3, **(model_kwargs["net"]["obs_enc"]["cfg"][obs_enc_name]), 
                last_activate=False,
            )
        elif obs_enc_name == "res":
            self.obs_enc = ResnetBackBone(
                **(model_kwargs["net"]["obs_enc"]["cfg"][obs_enc_name]),
                last_activate=False,
            )
        else:
            raise NotImplementedError(obs_enc_name)
        self.sta_enc = MLP(
            ROBOT_DIM, **(model_kwargs["net"]["sta_enc"]), 
            last_activate=False,
        )
        self.transformer = nn.Transformer(
            d_model=d_model,  batch_first=True,
            **(model_kwargs["net"]["transformer"]),
        )
        action_pred_dim = ROBOT_DIM - 2 + 6
        self.pred_mlp = MLP(
            input_dim=d_model, output_dim=action_pred_dim, last_activate=False,
            **(model_kwargs["net"]["mlp"]),
        )
        self.pe = PositionalEncoding(d_model=d_model)

        self.d_model = d_model
        self.action_pred_dim = action_pred_dim

        self.obs_horizon = int(model_kwargs["obs_horizon"])
        self.actpred_len = int(model_kwargs["actpred_len"])

        self.joints_unit = float(model_kwargs["joints_unit"])
        self.joints_weight = float(model_kwargs["joints_weight"])
        self.grip_weight = float(model_kwargs["grip_weight"])
        self.loss_str = str(learn_kwargs["loss"])

        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            params=[
                *self.obs_enc.parameters(), *self.sta_enc.parameters(), 
                *self.transformer.parameters(), *self.pred_mlp.parameters(),
            ],
            lr=self.learn_kwargs["optimizer"]["cfg"]["lr"],
            weight_decay=self.learn_kwargs["optimizer"]["cfg"]["weight_decay"],
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.learn_kwargs["scheduler"]["name"])(
            optimizer, **(self.learn_kwargs["scheduler"]["cfg"]),
        )
        '''warmup_step = 500
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer=optimizer, milestones=[warmup_step], 
            schedulers=[
                torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=0.01, end_factor=1.0, total_iters=warmup_step),
                torch.optim.lr_scheduler.MultiStepLR(optimizer, [10000 - warmup_step, 20000 - warmup_step], gamma=0.1)
            ]
        )'''
        return [optimizer], [scheduler]

    def forward_net(self, obs: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        (B, L, C, H, W), device, dtype = obs.shape, obs.device, obs.dtype

        obs_token: torch.Tensor = self.obs_enc(obs.view(B*L, C, H, W)) # B*L, FD, FH, FW
        _, D, FH, FW = obs_token.shape
        obs_token = obs_token.view(B*L, D, FH*FW).transpose(1, 2).reshape(B, L*FH*FW, D)

        state = state.view(B*L, -1)
        state = torch.concat([state[:, :2], state[:, 2:] / self.joints_unit], dim=1)
        sta_token: torch.Tensor = self.sta_enc(state).view(B, L, D)
        
        tsf_src = torch.concat([obs_token, sta_token], dim=1) # [B, L', D]
        tsf_tgt = torch.zeros((B, self.actpred_len, self.d_model), device=device, dtype=dtype) # [B, P, D]
        tsf_out = self.transformer(self.pe(tsf_src), self.pe(tsf_tgt)) # [B, P, D]

        action_pred = self.pred_mlp(tsf_out.view(-1, self.d_model)).view(B, self.actpred_len, self.action_pred_dim) # [B, P, A]
        # action_pred[:, :, 6:] *= self.joints_unit
        return action_pred
    
    def forward_all(self, batch: Dict[str, torch.Tensor]):
        action, state, obs = batch["action"], batch["state"], batch["obs"]
        action_pred = self.forward_net(obs, state)
        assert action_pred.shape[0] == action.shape[0]
        assert action_pred.shape[1] == action.shape[1]
        action_pred = action_pred.view(-1, self.action_pred_dim)
        action = action.view(-1, ROBOT_DIM)
        
        # loss_joints = getattr(nn.functional, self.loss_str)(action_pred[:, 6:] / self.joints_unit, action[:, 2:] / self.joints_unit)
        loss_joints = getattr(nn.functional, self.loss_str)(action_pred[:, 6:], action[:, 2:]) # [B*P, 7+7+4]
        loss_grip_l = nn.functional.cross_entropy(action_pred[:, 0:3], action[:, 0].to(dtype=torch.long))
        loss_grip_r = nn.functional.cross_entropy(action_pred[:, 3:6], action[:, 1].to(dtype=torch.long))

        loss_dict = dict(
            joints=loss_joints,
            grip_l=loss_grip_l,
            grip_r=loss_grip_r,
        )
        info_dict = dict(
            grip_l_err=(action_pred[:, 0:3].max(dim=1).indices != action[:, 0]).float().mean(),
            grip_r_err=(action_pred[:, 3:6].max(dim=1).indices != action[:, 1]).float().mean(),
            joints_err=(action_pred[:, 6:] - action[:, 2:]).abs().mean()
        )
        total_loss = loss_joints * self.joints_weight + (loss_grip_l + loss_grip_r) * self.grip_weight

        return total_loss, loss_dict, info_dict
    
    def log_all(self, total_loss: torch.Tensor, loss_dict: Dict[str, torch.Tensor], info_dict: Dict[str, torch.Tensor], name: str):
        self.log(f"{name}_loss/total_loss", total_loss.detach().clone(), sync_dist=True)
        for k, v in loss_dict.items():
            self.log(f"{name}_loss/{k}", v.detach().clone(), sync_dist=True)
        for k, v in info_dict.items():
            self.log(f"{name}_info/{k}", v.detach().clone(), sync_dist=True)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        sch = self.lr_schedulers()

        total_loss, loss_dict, info_dict = self.forward_all(batch)
        self.log_all(total_loss, loss_dict, info_dict, "train")
        
        opt.zero_grad()
        self.manual_backward(total_loss)
        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx):
        total_loss, loss_dict, info_dict = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(total_loss, loss_dict, info_dict, "valid")


class InsertPolicyACT(InsertPolicyE2E):
    def __init__(self, insert_gym: InsertGymE2E, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)

        self._module = InsertACTModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()
        if policy_cfg.action_w == "default":
            raise NotImplementedError
            self.action_w = 1.0 / self._module.actpred_len * (-1.) # different from origin paper which action_w is positive
        else:
            self.action_w = float(policy_cfg.action_w)
        logger.info(f"self.action_w {self.action_w}")

        self._obs_cache: List[np.ndarray] = []
        self._sta_cache: List[np.ndarray] = []
        self._act_cache: List[np.ndarray] = []

    def _prepare_net_input(self):
        obs_cache_len = len(self._obs_cache)
        assert obs_cache_len >= 1

        obs_input = []
        sta_input = []
        for obs_step in range(obs_cache_len - self._module.obs_horizon, obs_cache_len):
            obs_step = max(obs_step, 0)
            obs_raw = self._obs_cache[obs_step] # [3, B, H', W']
            C, B, H_, W_ = obs_raw.shape
            obs_raw = obs_raw.reshape(3*B, H_, W_).transpose(1, 2, 0) # [H', W', 3*B]
            obs_resized = cv2.resize(obs_raw, (self._module.model_kwargs["width"], self._module.model_kwargs["height"])) # [H, W, 3*B]
            obs_input.append(obs_resized.reshape(self._module.model_kwargs["height"], self._module.model_kwargs["width"], 3, B))
            sta_input.append(self._sta_cache[obs_step]) # [B, R]
            
        obs_input = np.array(obs_input) # [L, H, W, 3, B]
        obs_input = obs_input.transpose(4, 0, 3, 1, 2) # [B, L, C, H, W]
        obs_input = torch.tensor(obs_input, dtype=self.dtype, device=self.device).contiguous() # [B, L, C, H, W]

        sta_input = np.array(sta_input) # [L, B, R]
        sta_input = sta_input.transpose(1, 0, 2) # [B, L, R]
        sta_input = torch.tensor(sta_input, dtype=self.dtype, device=self.device).contiguous() # [B, L, R]

        return obs_input, sta_input
    
    def _inference_net(self, obs_input, sta_input):
        with torch.no_grad():
            action_pred = self._module.forward_net(obs_input, sta_input)
        return action_pred
    
    def _get_action_to_execute(self):
        action = []
        weight = []
        num_action_in_cache = len(self._act_cache)
        P = self._module.actpred_len
        for i in range(P):
            idx_action_in_cache = num_action_in_cache - P + i

            if idx_action_in_cache >= 0:
                action.append(self._act_cache[idx_action_in_cache][:, P - 1 - i, :])
                weight.append(np.exp(-self.action_w * i))
        
        weight = np.array(weight) / np.sum(weight)
        action = (np.array(action) * weight[:, None, None]).sum(axis=0) # [B, A]

        action = np.concatenate([
            np.argmax(action[:, 0:3], axis=1)[:, None], 
            np.argmax(action[:, 3:6], axis=1)[:, None], 
            action[:, 6:],
        ], axis=1, dtype=np.float32)
        return torch.tensor(action, dtype=self.dtype, device=self.device)

    def get_action(self):
        color_all, depth_all, mask_garment_all, mask_hanger_all = self._insert_gym.e2e_get_obs()
        self._obs_cache.append(
            np.array([
                np.clip(depth_all, None, MAX_DEPTH) / MAX_DEPTH, 
                mask_garment_all, mask_hanger_all
            ])
        )

        state = self._insert_gym.e2e_get_state()
        self._sta_cache.append(utils.torch_to_numpy(state))

        obs_input, sta_input = self._prepare_net_input()
        action_pred = self._inference_net(obs_input, sta_input) # [B, P, A]
        self._act_cache.append(utils.torch_to_numpy(action_pred))

        # os.makedirs("test_act", exist_ok=True)
        # np.save(f"test_act/{len(self._act_cache) - 1}.npy", self._act_cache[-1])

        # os.makedirs("test_sta", exist_ok=True)
        # np.save(f"test_sta/{len(self._sta_cache) - 1}.npy", self._sta_cache[-1])

        # os.makedirs("test_obs", exist_ok=True)
        # np.save(f"test_obs/{len(self._obs_cache) - 1}.npy", self._obs_cache[-1])

        action_to_exe = self._get_action_to_execute()
        return action_to_exe
    
    def reset(self):
        self._obs_cache = []
        self._sta_cache = []
        self._act_cache = []