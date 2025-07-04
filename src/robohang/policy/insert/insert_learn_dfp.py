import logging
logger = logging.getLogger(__name__)

import os
import copy
from typing import List, Dict, Literal, Union, Tuple, Any, Optional
import json
import math
from dataclasses import dataclass
import pprint
from queue import Queue

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
from multiprocessing import Manager
import cv2
import einops

import tqdm
import matplotlib.pyplot as plt
import matplotlib

import trimesh.transformations as tra
import trimesh
import omegaconf

import batch_urdf
import robohang.common.utils as utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.insert.net_e2e import CNNBackBone, ResnetBackBone, MLP, PositionalEncoding, CNN
from robohang.policy.insert.net_dfp import ConditionalUnet1D, TransformerForDiffusion
from robohang.policy.insert.insert_gym_e2e import ROBOT_DIM # Action dimension
from robohang.policy.insert.insert_policy_e2e import InsertPolicyE2E, InsertGymE2E

import lightning.pytorch as pl
from diffusers import DDPMScheduler, UNet2DModel


MAX_DEPTH = 2.0
@dataclass
class TrajectoryInfo:
    path: str
    action_num: int


# normalizer
DEFAULT_SCALE_MULTIPLE = 1.2

def get_scale(vmin: np.ndarray, vmax: np.ndarray, use_max_scale: bool, scale_multiply=DEFAULT_SCALE_MULTIPLE):
    v1, v2 = np.min([vmin, vmax], axis=0), np.max([vmin, vmax], axis=0)
    scale = (v2 - v1) / 2 * scale_multiply + 1e-6
    if use_max_scale:
        scale = np.max(scale)
    return scale

def normalize(x: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, forward: bool, use_max_scale: bool, scale_multiply=DEFAULT_SCALE_MULTIPLE):
    offset = (vmin + vmax) / 2
    scale = get_scale(vmin, vmax, use_max_scale, scale_multiply)
    if forward:
        return (x - offset) / scale
    else:
        return x * scale + offset


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
        
        # stat
        self.normalize_use_max_scale = bool(ds_cfg.normalize_use_max_scale)
        self.stat = np.load(utils.get_path_handler()(self._ds_cfg.statistics), allow_pickle=True).item()

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
                a_raw = np.load(path).astype(self._dtype)

                # onehot and preprocess
                a = np.zeros((ROBOT_DIM + 4, ), dtype=self._dtype)
                a[6:] = normalize(
                    a_raw[2:], 
                    self.stat["action"]["min"][2:], self.stat["action"]["max"][2:], 
                    True, self.normalize_use_max_scale
                )
                a[int(a_raw[0])] = 1.
                a[int(a_raw[1]) + 3] = 1.
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


class InsertDfPModule(pl.LightningModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.obs_horizon = int(model_kwargs["obs_horizon"])
        self.actpred_len = int(model_kwargs["actpred_len"])
        self.normalize_use_max_scale = bool(model_kwargs["normalize_use_max_scale"])

        self.joints_weight = float(model_kwargs["joints_weight"])
        self.grip_weight = float(model_kwargs["grip_weight"])
        self.joint_loss_str = str(learn_kwargs.get("joint_loss_str", "mse_loss"))

        # net
        self.net_name: Literal["cnn", "transformer"] = str(model_kwargs["net"]["name"])
        if self.net_name == "none":  # TODO: TMP solution
            self.net_name = "cnn"
            net_cfg = model_kwargs["net"]
        else:
            net_cfg = model_kwargs["net"][self.net_name]
        if self.net_name == "cnn":
            self.enc_out_dim = net_cfg["enc_out_dim"]
            self.obs_enc = CNN(
                input_height=model_kwargs["height"], input_width=model_kwargs["width"], 
                input_channel=3, output_channel=self.enc_out_dim, **(net_cfg["obs_enc"]),
                last_activate=False,
            )
            self.sta_enc = MLP(
                input_dim=ROBOT_DIM, output_dim=self.enc_out_dim, **(net_cfg["sta_enc"]), 
                last_activate=False,
            )
            self.cond_unet = ConditionalUnet1D(
                input_dim=ROBOT_DIM + 4, 
                global_cond_dim=self.enc_out_dim * 2 * self.obs_horizon,
                **(net_cfg["cond_unet"])
            )
        elif self.net_name == "transformer":
            self.obs_enc = CNNBackBone(
                input_channel=3, input_height=model_kwargs["height"], input_width=model_kwargs["width"],
                last_activate=False, **(net_cfg["obs_enc"])
            )
            self.sta_enc = MLP(
                input_dim=ROBOT_DIM, **(net_cfg["sta_enc"]), 
                last_activate=False,
            )
            self.transformer = TransformerForDiffusion(
                input_dim=ROBOT_DIM + 4, output_dim=ROBOT_DIM + 4, 
                horizon=self.actpred_len, n_obs_steps=(self.obs_enc.FH * self.obs_enc.FW + 1) * self.obs_horizon,
                cond_dim=net_cfg["token_dim"], causal_attn=True, obs_as_cond=True, 
                **(net_cfg["transformer"]), 
            )
        else:
            raise NotImplementedError(self.net_name)

        # ddpm
        self.ddpm = DDPMScheduler(**model_kwargs["ddpm"])

        # stat
        self.stat = np.load(utils.get_path_handler()(model_kwargs["stat_path"]), allow_pickle=True).item()

        self.automatic_optimization = False

    def configure_optimizers(self):
        if self.net_name == "cnn":
            params = [*self.obs_enc.parameters(), *self.sta_enc.parameters(), *self.cond_unet.parameters()]
        elif self.net_name == "transformer":
            params = [*self.obs_enc.parameters(), *self.sta_enc.parameters(), *self.transformer.parameters()]
        else:
            raise NotImplementedError(self.net_name)
        optimizer = torch.optim.AdamW(
            params=params,
            lr=self.learn_kwargs["optimizer"]["cfg"]["lr"],
            weight_decay=self.learn_kwargs["optimizer"]["cfg"]["weight_decay"],
        )
        scheduler = getattr(torch.optim.lr_scheduler, self.learn_kwargs["scheduler"]["name"])(
            optimizer, **(self.learn_kwargs["scheduler"]["cfg"]),
        )
        return [optimizer], [scheduler]

    def forward_diffusion(self, a: torch.Tensor, t: torch.Tensor):
        """
        Args:
        - a: (B, T, D)
        - t: (B, )

        Return:
        - a_noised: (B, T, D)
        - noise: (B, T, D)
        """
        noise = torch.randn_like(a, device=a.device)
        a_noised = self.ddpm.add_noise(a, noise, t)
        return a_noised, noise
    
    def reverse_diffusion(self, obs: torch.Tensor, sta: torch.Tensor, a: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        (B, L, C, H, W), device, dtype = obs.shape, obs.device, obs.dtype

        obs_token: torch.Tensor = self.obs_enc(obs.view(B*L, C, H, W)) # B*L, D
        sta_token: torch.Tensor = self.sta_enc(sta.view(B*L, -1)) # B*L, D

        if self.net_name == "cnn":
            noise_pred = self.cond_unet(
                sample=a, timestep=t, 
                global_cond=torch.concat([obs_token.view(B, -1), sta_token.view(B, -1)], dim=1)
            )
        elif self.net_name == "transformer":
            obs_token: torch.Tensor = self.obs_enc(obs.view(B*L, C, H, W))
            _, D, FH, FW = obs_token.shape
            obs_token = obs_token.view(B*L, D, FH*FW).transpose(1, 2).reshape(B, L*FH*FW, D)
            sta_token: torch.Tensor = self.sta_enc(sta.view(B*L, -1)).view(B, L, D)

            noise_pred = self.transformer(
                sample=a, timestep=t, 
                cond=torch.concat([obs_token, sta_token], dim=1)
            )
        else:
            raise NotImplementedError(self.net_name)
        return noise_pred
    
    def inference_ddpm(self, obs: torch.Tensor, sta: torch.Tensor, ddpm_inference_timestep: int=None, use_tqdm=False) -> torch.Tensor:
        (B, L, C, H, W), device, dtype = obs.shape, obs.device, obs.dtype
        trajectory = torch.randn(
            (B, self.actpred_len, ROBOT_DIM + 4), device=device, dtype=dtype,
        )
        if ddpm_inference_timestep is not None:
            self.ddpm.set_timesteps(ddpm_inference_timestep)
        with torch.no_grad():
            # assume inference timesteps is the same as the training timesteps
            for t in (tqdm.tqdm(self.ddpm.timesteps) if use_tqdm else self.ddpm.timesteps):
                model_output = self.reverse_diffusion(obs, sta, trajectory, t)
                trajectory = self.ddpm.step(model_output, t, trajectory).prev_sample
        return trajectory
    
    def get_joints_scale(self, dtype, device):
        return torch.tensor(get_scale(
            self.stat["action"]["min"][2:], self.stat["action"]["max"][2:], self.normalize_use_max_scale
        ), dtype=dtype, device=device)
    
    def forward_all(self, batch: Dict[str, torch.Tensor]):
        a, sta, obs = batch["action"], batch["state"], batch["obs"]
        (B, T, DA), device, dtype = a.shape, a.device, a.dtype

        timesteps = torch.randint(0, self.ddpm.config.num_train_timesteps, (B, ), device=device, dtype=torch.long)
        a_noised, noise = self.forward_diffusion(a, timesteps)
        noise_pred = self.reverse_diffusion(obs, sta, a_noised, timesteps)

        assert noise.shape == (B, T, ROBOT_DIM + 4), noise.shape
        assert noise_pred.shape == (B, T, ROBOT_DIM + 4), noise_pred.shape

        loss_joints = getattr(nn.functional, self.joint_loss_str)(noise_pred[:, :, 6:], noise[:, :, 6:])
        loss_grip_l = nn.functional.mse_loss(noise_pred[:, :, 0:3], noise[:, :, 0:3])
        loss_grip_r = nn.functional.mse_loss(noise_pred[:, :, 3:6], noise[:, :, 3:6])

        loss_dict = dict(
            joints=loss_joints,
            grip_l=loss_grip_l,
            grip_r=loss_grip_r,
        )
        info_dict = dict(
            joints_err=((noise_pred[:, :, 6:] - noise[:, :, 6:]) * self.get_joints_scale(dtype, device)).abs().mean(),
            grip_l_err=(noise_pred[:, :, 0:3] - noise[:, :, 0:3]).abs().mean(),
            grip_r_err=(noise_pred[:, :, 3:6] - noise[:, :, 3:6]).abs().mean(),
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


class InsertPolicyDfP(InsertPolicyE2E):
    def __init__(self, insert_gym: InsertGymE2E, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)

        self._module = InsertDfPModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()
        self.ddpm_inference_timestep = int(policy_cfg.ddpm_inference_timestep)
        self.actpred_len = int(self._module.actpred_len)

        self.use_time_ensemble = bool(policy_cfg.use_time_ensemble)
        self.act_exe_len = int(policy_cfg.act_exe_len)
        logger.info(f"self.act_exe_len {self.act_exe_len}")
        self.action_w = float(policy_cfg.action_w)
        logger.info(f"self.action_w {self.action_w}")

        self._obs_cache: List[np.ndarray] = []
        self._sta_cache: List[np.ndarray] = []
        self._act_cache: List[np.ndarray] = []
        self._act_queue: Queue[np.ndarray] = Queue()

        # self.debug_idx = 0 # DEBUG

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
    
    def _inference_net(self, obs_input: torch.Tensor, sta_input: torch.Tensor):
        action_pred = self._module.inference_ddpm(
            obs_input, sta_input, ddpm_inference_timestep=self.ddpm_inference_timestep,
            use_tqdm=False,
        )
        assert action_pred.shape == (self.batch_size, self.actpred_len, ROBOT_DIM + 4)
        return action_pred
    
    def _post_process_action(self, action_normalized: np.ndarray):
        action = np.zeros((self.batch_size, ROBOT_DIM))

        # os.makedirs("test_act", exist_ok=True) # DEBUG
        # np.save(f"test_act/{self.debug_idx}.npy", action_normalized) # DEBUG
        # self.debug_idx += 1 # DEBUG

        action[:, 0] = np.argmax(action_normalized[:, 0:3], axis=1)
        action[:, 1] = np.argmax(action_normalized[:, 3:6], axis=1)
        action[:, 2:] = normalize(
            action_normalized[:, 6:], 
            self._module.stat["action"]["min"][2:], self._module.stat["action"]["max"][2:], 
            False, self._module.normalize_use_max_scale
        )

        return torch.tensor(action, dtype=self.dtype, device=self.device)

    def _get_action_to_execute_with_ensemble(self):
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
        return action
    
    def get_action(self):
        if not self.use_time_ensemble:
            if self._act_queue.empty():
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
                action_pred_np = utils.torch_to_numpy(action_pred)
                for p in range(self.act_exe_len):
                    self._act_queue.put(action_pred_np[:, p, :])
            
            action_exe = self._act_queue.get()
            action = self._post_process_action(action_exe)
            return action
        else:
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

            action_exe = self._get_action_to_execute_with_ensemble()
            action = self._post_process_action(action_exe)
            return action

    
    def reset(self):
        self._obs_cache = []
        self._sta_cache = []
        self._act_cache = []
        self._act_queue = Queue()