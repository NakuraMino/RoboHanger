import os
import copy
from typing import List, Dict, Literal, Union, Tuple, Any, Optional
import json
import math
from dataclasses import dataclass

import matplotlib.figure
import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from multiprocessing import Manager
import concurrent.futures
import cv2

import tqdm
import matplotlib.pyplot as plt
import matplotlib

import trimesh.transformations as tra
import trimesh
import omegaconf

import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
import robohang.policy.policy_utils as policy_utils
from robohang.policy.net import UNet
import robohang.common.utils as utils

import lightning.pytorch as pl


# helper classes
class FunnelScoreCalculator:
    @staticmethod
    def score_single_frame(
        score_batch,
        batch_idx,
    ) -> float:
        # coverage / canonicalization
        return float(score_batch["coverage"][batch_idx])

    @staticmethod
    def reward(
        score_before: float, 
        score_after: float,
    ) -> float:
        return score_after - score_before


class FunnelActEncDec:
    enc_dtype = np.float32
    enc_shape = (4, )

    fling_center_ij_idx = slice(0, 2)
    fling_distance_idx = 2
    fling_angle_degree_idx = 3

    fling_direct_ij_left_idx = slice(0, 2)
    fling_direct_ij_right_idx = slice(2, 4)

    @classmethod
    def encode(self, act_batch: dict, batch_idx: int, depth_img: np.ndarray):
        """pack action and call `xyz2ij`"""
        act = np.zeros(shape=FunnelActEncDec.enc_shape, dtype=FunnelActEncDec.enc_dtype)
        action_direct_ij = np.zeros(shape=(4, ), dtype=int)
        z = MetaInfo.calculate_z_world(depth_img)

        cxy = act_batch["center_xy"]
        dist = act_batch["distance"]
        ang_deg = act_batch["angle_degree"]

        act[self.fling_center_ij_idx] = policy_utils.xyz2ij(
            cxy[batch_idx, 0],
            cxy[batch_idx, 1],
            z, MetaInfo.reproject_camera_info
        )
        act[self.fling_distance_idx] = dist[batch_idx]
        act[self.fling_angle_degree_idx] = ang_deg[batch_idx]

        action_direct_ij[self.fling_direct_ij_left_idx] = policy_utils.xyz2ij(
            cxy[batch_idx, 0] - dist[batch_idx] * math.cos(math.radians(ang_deg[batch_idx])) / 2,
            cxy[batch_idx, 1] - dist[batch_idx] * math.sin(math.radians(ang_deg[batch_idx])) / 2,
            z, MetaInfo.reproject_camera_info
        )
        action_direct_ij[self.fling_direct_ij_right_idx] = policy_utils.xyz2ij(
            cxy[batch_idx, 0] + dist[batch_idx] * math.cos(math.radians(ang_deg[batch_idx])) / 2,
            cxy[batch_idx, 1] + dist[batch_idx] * math.sin(math.radians(ang_deg[batch_idx])) / 2,
            z, MetaInfo.reproject_camera_info
        )
        
        return act, action_direct_ij
    
    @classmethod
    def decode_fling_ij(self, act_batch: torch.Tensor):
        """
        Args:
            act_batch: [B, 4]
        Return:
            ij: [B, 2], long
        """
        return act_batch[:, self.fling_center_ij_idx].to(torch.long)
    
    @classmethod
    def decode_fling_direct_ij(self, direct_ij_batch: torch.Tensor, name: Literal["left", "right"]):
        """
        Args:
            direct_ij_batch: [B, 4]
        Return:
            ij: [B, 2], long
        """
        if name == "left":
            return direct_ij_batch[:, self.fling_direct_ij_left_idx].to(torch.long)
        elif name == "right":
            return direct_ij_batch[:, self.fling_direct_ij_right_idx].to(torch.long)
        else:
            raise ValueError(name)

    @classmethod
    def transform_image_and_action(
        self, 
        distance_range: Tuple[float, float],
        depth: np.ndarray, 
        mask: np.ndarray, 
        action_space: np.ndarray,
        action_angle_degree: float,
        action_distance: float,
        action_center_ij: np.ndarray=None,
    ):
        # assert action.shape == self.enc_shape, action.shape
        ret = dict()

        # rotation center
        H, W = depth.shape
        center = (W // 2, H // 2)

        # rotation scale
        scale = min(*distance_range) / action_distance

        # rotation angle
        angle = -action_angle_degree

        # rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)

        # apply rotation and scale
        depth = cv2.warpAffine(depth, rotation_matrix, (W, H), borderMode=cv2.BORDER_REPLICATE)
        mask = cv2.warpAffine(mask, rotation_matrix, (W, H))
        action_space = cv2.warpAffine(action_space, rotation_matrix, (W, H), flags=cv2.INTER_NEAREST)
        
        ret["depth"] = depth
        ret["mask"] = mask
        ret["action_space"] = action_space
        if action_center_ij is not None:
            j, i = rotation_matrix @ np.array([action_center_ij[1], action_center_ij[0], 1.], dtype=float)
            action_center_ij = np.array([i, j])
            ret["action_center_ij"] = action_center_ij

        return ret


def rotate_and_translate(
    aug_cfg: omegaconf.DictConfig, 
    depth: np.ndarray, 
    mask: np.ndarray, 
    ij_left: Optional[np.ndarray]=None, 
    ij_right: Optional[np.ndarray]=None, 
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    # rotation matrix
    H, W = depth.shape
    center = np.array((W // 2, H // 2), dtype=float) + utils.map_01_ab(np.random.random(2), -aug_cfg.translate, +aug_cfg.translate)
    angle = utils.map_01_ab(np.random.random(), 0., 360.)
    scale = utils.map_01_ab(np.random.random(), aug_cfg.scale[0], aug_cfg.scale[1])
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    
    # apply rotation and scale
    depth = cv2.warpAffine(depth, rotation_matrix, (W, H), borderMode=cv2.BORDER_REPLICATE)
    mask = cv2.warpAffine(mask, rotation_matrix, (W, H))

    def get_new_ij(ij: Optional[np.ndarray]):
        if ij is not None:
            j, i = rotation_matrix @ np.array([ij[1], ij[0], 1.], dtype=float)
            return np.array([i, j], dtype=float)

    return depth, mask, get_new_ij(ij_left), get_new_ij(ij_right)


# dataset
@dataclass
class FunnelDataDict:
    depth_path: str
    mask_path: str
    depth_next_path: str
    mask_next_path: str
    
    reward: float
    reward_cov: float
    reward_ali: float
    action: np.ndarray
    action_direct_ij: np.ndarray


class FunnelDataset(Dataset):
    def __init__(
        self,
        data_list: List[FunnelDataDict], 
        data_index_table: np.ndarray, 
        ds_cfg: omegaconf.DictConfig,
        name="none",
    ) -> None:
        super().__init__()

        self._data_list = Manager().list(data_list)
        self._data_index_table = data_index_table.copy()
        self._ds_cfg = copy.deepcopy(ds_cfg)

        self._size = int(len(self._data_index_table))
        self._dtype = getattr(np, self._ds_cfg.dtype)

        self._act_str = "fn"

        print(f"funnel dataset {name}: len {self._size}")
    
    def __len__(self):
        return self._size
    
    def get_angle_range_list(self) -> List[float]:
        return np.linspace(
            min(MetaInfo.fling_normal_action_space.angle_degree),
            max(MetaInfo.fling_normal_action_space.angle_degree),
            self._ds_cfg.angle_degree_num,
        ).tolist()
    
    def get_distance_list(self) -> List[float]:
        return np.linspace(
            min(MetaInfo.fling_normal_action_space.distance),
            max(MetaInfo.fling_normal_action_space.distance),
            self._ds_cfg.distance_num,
        ).tolist()
        
    def __getitem__(self, index: int):
        # extract data
        data: FunnelDataDict = self._data_list[self._data_index_table[index]]
        depth_raw: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_raw: np.ndarray = np.load(data.mask_path).astype(self._dtype)
        depth_next_raw: np.ndarray = np.load(data.depth_next_path).astype(self._dtype)
        mask_next_raw: np.ndarray = np.load(data.mask_next_path).astype(self._dtype)

        action_raw: np.ndarray = data.action.copy().astype(self._dtype)
        action_direct_ij_raw: np.ndarray = data.action_direct_ij.copy().astype(self._dtype)

        # make action space mask and positional encoding
        action_space_raw = np.zeros_like(depth_raw)
        i, j = MetaInfo.get_action_space_slice(self._act_str, depth_raw)
        action_space_raw[i, j] = 1.

        # copy action
        action = action_raw.copy()
        action_space = action_space_raw.copy()

        # augment data
        depth, mask, info = learn_utils.data_augmentation(depth_raw, mask_raw, self._ds_cfg.aug)

        # transform img
        result = FunnelActEncDec.transform_image_and_action(
            distance_range=MetaInfo.fling_normal_action_space.distance,
            depth=depth,
            mask=mask,
            action_space=action_space_raw,
            action_angle_degree=action[FunnelActEncDec.fling_angle_degree_idx],
            action_distance=action[FunnelActEncDec.fling_distance_idx],
            action_center_ij=action[FunnelActEncDec.fling_center_ij_idx],
        )
        depth, mask, action_space, action_center_ij = result["depth"], result["mask"], result["action_space"], result["action_center_ij"]
        action = action.copy()
        action[FunnelActEncDec.fling_center_ij_idx] = action_center_ij

        # transform img for next obs
        depth_next = []
        mask_next = []
        action_space_next = []
        for a in self.get_angle_range_list():
            for d in self.get_distance_list():
                result = FunnelActEncDec.transform_image_and_action(
                    distance_range=MetaInfo.fling_normal_action_space.distance,
                    depth=depth_next_raw,
                    mask=mask_next_raw,
                    action_space=action_space_raw,
                    action_angle_degree=a,
                    action_distance=d,
                )
                depth_next.append(result["depth"])
                mask_next.append(result["mask"])
                action_space_next.append(result["action_space"])
        
        depth_next = np.array(depth_next)
        mask_next = np.array(mask_next)
        action_space_next = np.array(action_space_next)

        return dict(
            depth=depth,
            mask=mask,
            action=action,
            reward=np.float32(data.reward),
            action_space=action_space,
            
            depth_raw=depth_raw,
            mask_raw=mask_raw,
            action_raw=action_raw,
            action_space_raw=action_space_raw,
            action_direct_ij_raw=action_direct_ij_raw,

            depth_next=depth_next,
            mask_next=mask_next,
            action_space_next=action_space_next,

            depth_next_raw=depth_next_raw,
            mask_next_raw=mask_next_raw,
            reward_cov=np.float32(data.reward_cov),
            reward_ali=np.float32(data.reward_ali),
            depth_raw_path=os.path.relpath(data.depth_path, utils.get_path_handler()(".")),
        )


def make_funnel_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    make_cfg: omegaconf.DictConfig,
    ds_cfg: omegaconf.DictConfig,
):
    pattern = "^completed.txt$"
    df = learn_utils.DataForest(data_path, [pattern])
    node_n_raw = df.get_forest_size(pattern)

    all_data_list = Manager().list([])

    def worker(idx: int):
        misc_path = os.path.join(os.path.dirname(df.get_item(pattern, idx).file_path), "misc.json")
        with open(misc_path, "r") as f_obj:
            misc_info = json.load(f_obj)
        base_dir = os.path.dirname(misc_path)

        def format_step(step_idx):
            return str(step_idx).zfill(len(str(2 * misc_info["num_trial"])))
        
        def format_batch(batch_idx):
            return str(batch_idx).zfill(len(str(misc_info["batch_size"] - 1)))
        
        for trial_idx in range(int(misc_info["num_trial"])):
            act_batch = np.load(os.path.join(base_dir, "action", format_step(trial_idx * 2) + ".npy"), allow_pickle=True).item()
            with open(os.path.join(base_dir, "score", format_step(trial_idx * 2) + ".json"), "r") as f_obj:
                score_batch_before = json.load(f_obj)
            with open(os.path.join(base_dir, "score", format_step(trial_idx * 2 + 1) + ".json"), "r") as f_obj:
                score_batch_after = json.load(f_obj)
            with open(os.path.join(base_dir, "sim_error", format_step(trial_idx * 2 + 1) + ".json"), "r") as f_obj:
                sim_error_batch_after = json.load(f_obj)
            for batch_idx in range(int(misc_info["batch_size"])):
                if sim_error_batch_after["all"][batch_idx]: # skip sim_error batch
                    continue

                # obs
                obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))
                depth_path = os.path.join(obs_base_dir, "reproject_depth", format_step(trial_idx * 2) + ".npy")
                depth_next_path = os.path.join(obs_base_dir, "reproject_depth", format_step(trial_idx * 2 + 1) + ".npy")
                if bool(make_cfg.mask_interp):
                    mask_path = os.path.join(obs_base_dir, "reproject_is_garment", format_step(trial_idx * 2) + ".npy")
                    mask_next_path = os.path.join(obs_base_dir, "reproject_is_garment", format_step(trial_idx * 2 + 1) + ".npy")
                else:
                    mask_path = os.path.join(obs_base_dir, "reproject_is_garment_nointerp", format_step(trial_idx * 2) + ".npy")
                    mask_next_path = os.path.join(obs_base_dir, "reproject_is_garment_nointerp", format_step(trial_idx * 2 + 1) + ".npy")

                with open(os.path.join(obs_base_dir, "info", format_step(trial_idx * 2) + ".json"), "r") as f_obj:
                    obs_info = json.load(f_obj)
                assert MetaInfo.check_reproject_info(obs_info["reproject"]), str(obs_info["reproject"])
                
                # reward
                score_before = FunnelScoreCalculator.score_single_frame(score_batch_before, batch_idx)
                score_after = FunnelScoreCalculator.score_single_frame(score_batch_after, batch_idx)
                kp_path = {
                    lr_str: [
                        os.path.join(base_dir, "keypoints", stp_str, lr_str, format_batch(batch_idx) + ".npy")
                        for stp_str in [format_step(trial_idx * 2), format_step(trial_idx * 2 + 1)]
                    ] for lr_str in ["left", "right"]
                }
                def kp_smooth_func(kp_img: np.ndarray):
                    return float(np.clip((kp_img > 0.5).astype(np.float32).sum() / float(make_cfg.kp_smooth_pixel), 0., 1.))
                kp_before = kp_smooth_func(np.load(kp_path["left"][0])) * kp_smooth_func(np.load(kp_path["right"][0]))
                kp_after = kp_smooth_func(np.load(kp_path["left"][1])) * kp_smooth_func(np.load(kp_path["right"][1]))
                
                reward_cov = FunnelScoreCalculator.reward(score_before, score_after)
                reward_ali = kp_after - kp_before
                reward = reward_cov + reward_ali * float(make_cfg.kp_weight)

                # action
                action, action_direct_ij = FunnelActEncDec.encode(act_batch, batch_idx, np.load(depth_path))

                # assemble data
                data_dict = FunnelDataDict(
                    depth_path=depth_path,
                    mask_path=mask_path,
                    depth_next_path=depth_next_path,
                    mask_next_path=mask_next_path,
                    reward=reward,
                    reward_cov=reward_cov,
                    reward_ali=reward_ali,
                    action=action,
                    action_direct_ij=action_direct_ij,
                )

                all_data_list.append(data_dict)
    
    print("scanning all trajectories ...")
    launcher = learn_utils.MultiProcessLauncher(make_cfg.num_worker)
    for idx in tqdm.tqdm(range(node_n_raw)):
        launcher.launch_worker(worker, args=(idx, ))
    launcher.join_all()
    
    total_size = len(all_data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    trds = FunnelDataset(all_data_list, path_idx_permutated[valid_size:], ds_cfg, name=f"train")
    vlds = FunnelDataset(all_data_list, path_idx_permutated[:valid_size], ds_cfg, name=f"valid")
    return trds, vlds


class ValueNet:
    def __init__(self, ckpt: str):
        from robohang.policy.funnel.keypoints_unet import KeypointsModule
        
        if isinstance(ckpt, str):
            self.net = KeypointsModule.load_from_checkpoint(
                checkpoint_path=utils.get_path_handler()(ckpt),
                map_location=torch.device("cpu"),
            ).eval()
    
    def to_device(self, device):
        if hasattr(self, "net"):
            self.net.to(device)
    
    def _calculate_score(self, pred_l: torch.Tensor, pred_r: torch.Tensor):
        return torch.sigmoid(pred_l) * torch.sigmoid(pred_r)
    
    def predict_reward(self, depth: torch.Tensor, mask: torch.Tensor, depth_next: torch.Tensor, mask_next: torch.Tensor):
        if hasattr(self, "net"):
            i = self.net.preprocess_input(depth, mask)
            pred = self.net.run_predict_cls(i)
            i_next = self.net.preprocess_input(depth_next, mask_next)
            pred_next = self.net.run_predict_cls(i_next)
            reward = torch.sigmoid(pred_next) - torch.sigmoid(pred)
            assert reward.shape == (depth.shape[0], ), f"{reward.shape}, {depth.shape}"
        else:
            reward = torch.zeros(depth.shape[0], device=depth.device, dtype=depth.dtype)
        return reward


class FunnelFlingNormalModule(pl.LightningModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.d_unit = float(model_kwargs["d_unit"])
        self.discount = float(learn_kwargs["rl"]["discount"])
        self.double_q = bool(learn_kwargs["rl"]["double_q"])
        self.net_target_update_freq = int(learn_kwargs["rl"]["net_target_update_freq"])
        self.automatic_optimization = False

        kwargs = copy.deepcopy(model_kwargs["net"])
        kwargs["input_channel"] = 2
        kwargs["output_channel"] = 1
        self.net = UNet(**kwargs)
        if self.double_q:
            self.net_target = UNet(**kwargs)
        
        '''self.val_net = ValueNet(model_kwargs["val_net"]["ckpt"])
        self.val_net_weight = float(model_kwargs["val_net"]["weight"])'''
    
    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
                self.net.parameters(), 
                **(self.learn_kwargs["optimizer"]["cfg"]),
            )
        ]
        schedule = [
            getattr(torch.optim.lr_scheduler, self.learn_kwargs["schedule"]["name"])(
                optimizer[0],
                **(self.learn_kwargs["schedule"]["cfg"]),
            )
        ]
        return optimizer, schedule
    
    def normalize_depth(self, depth: torch.Tensor):
        B = depth.shape[0]
        return (depth - depth.view(B, -1).mean(-1).view(B, 1, 1)) / self.d_unit
    
    def preprocess_input(self, depth: torch.Tensor, mask: torch.Tensor):
        """
        `d = (d - mean) / unit`

        Args:
        - depth: [B, H, W]
        - mask: [B, H, W]

        Return:
        - i: [B, 2, H, W]
        """
        B, H, W = depth.shape
        depth = self.normalize_depth(depth)
        return torch.concat([depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)

    def extract_data(self, batch):
        """
        Return:
        - depth: [B, H, W]
        - mask: [B, H, W]
        - action: [B, 4]
        - reward: [B]

        - depth_next: [B, S, H, W]
        - mask_next: [B, S, H, W]
        - action_space_next: [B, S, H, W]
        """
        depth: torch.Tensor = batch["depth"] # [B, H, W]
        mask: torch.Tensor = batch["mask"] # [B, H, W]
        action: torch.Tensor = batch["action"] # [B, 4]

        B, H, W = depth.shape
        assert mask.shape == depth.shape, mask.shape
        assert action.shape == (B, 4), action.shape

        reward: torch.Tensor = batch["reward"] # [B]
        assert reward.shape == (B, ), reward.shape

        '''reward_cov: torch.Tensor = batch["reward"] # [B]
        assert reward_cov.shape == (B, ), reward_cov.shape'''

        depth_next: torch.Tensor = batch["depth_next"] # [B, S, H, W]
        mask_next: torch.Tensor = batch["mask_next"] # [B, S, H, W]
        action_space_next: torch.Tensor = batch["action_space_next"] # [B, S, H, W]

        B, S, H, W = depth_next.shape
        assert mask_next.shape == (B, S, H, W), mask_next.shape
        assert action_space_next.shape == (B, S, H, W), action_space_next.shape

        return depth, mask, action, reward, depth_next, mask_next, action_space_next

        '''self.val_net.to_device(self.device)
        reward_key = self.val_net.predict_reward(depth, mask, depth_next, mask_next) * self.val_net_weight
        reward = (reward_cov + reward_key)
        assert reward.shape == (B, ), reward.shape
        return depth, mask, action, reward, depth_next, mask_next, action_space_next, reward_cov, reward_key'''

    def run_forward(
        self, 
        i: torch.Tensor, 
        a: torch.Tensor, 
    ) -> List[torch.Tensor]:
        """
        Args:
        - i: [B, C, H, W], float
        - a: [B, 2], long

        Return:
        - pred: [B]
        - dense: [B, H, W]
        """
        device = i.device
        o: torch.Tensor = self.net(i)
        assert len(o.shape) == 4 and o.shape[1] == 1, o.shape
        B, _, H, W = o.shape

        dense = o.squeeze(dim=1) # [B, H, W]
        pred = dense[torch.arange(B).to(device), a[:, 0], a[:, 1]]

        assert pred.shape == (B, ), pred.shape
        assert dense.shape == (B, H, W), dense.shape
        
        return [pred, dense]
    
    def calculate_label(self, reward: torch.Tensor, depth_next: torch.Tensor, mask_next: torch.Tensor, action_space_next: torch.Tensor):
        # preprocess
        B, S, H, W = depth_next.shape
        device = reward.device
        depth_next = depth_next.view((B*S, H, W))
        mask_next = mask_next.view((B*S, H, W))
        action_space_next = action_space_next.view((B*S, H, W))

        # calculate label
        i_next = self.preprocess_input(depth_next, mask_next)
        if self.double_q:
            action_next = learn_utils.out_of_action_space_to_min(
                self.net(i_next).squeeze(dim=1), action_space_next
            ).view(B, -1).max(dim=1).indices
            label: torch.Tensor = learn_utils.out_of_action_space_to_min(
                self.net_target(i_next).squeeze(dim=1), action_space_next
            ).view(B, -1)[torch.arange(B).to(device), action_next] * self.discount + reward
        else:
            label: torch.Tensor = learn_utils.out_of_action_space_to_min(
                self.net(i_next).squeeze(dim=1), action_space_next
            ).view(B, -1).max(dim=1).values * self.discount + reward
        assert label.shape == (B, ), label.shape
        return label
    
    def forward_all(self, batch):
        depth, mask, action, reward, depth_next, mask_next, action_space_next = self.extract_data(batch)

        # prepare label
        self.net.eval()
        if self.double_q:
            self.net_target.eval()
        label = self.calculate_label(reward, depth_next, mask_next, action_space_next).detach()

        # forward net
        self.net.train()
        a = FunnelActEncDec.decode_fling_ij(action) # [B, 2]
        i = self.preprocess_input(depth, mask)
        pred, dense = self.run_forward(i=i, a=a)
        
        loss: torch.Tensor = torch.mean(((pred - label) ** 2)) # scalar
        err: torch.Tensor = torch.mean(torch.abs(pred - label)) # scalar

        dense_info = dict(
            dense=dense,

            depth=depth,
            mask=mask,
            action=action,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],

            depth_next_raw=batch["depth_next_raw"],
            action_raw=batch["action_raw"],
            action_direct_ij_raw=batch["action_direct_ij_raw"],
            action_space=batch["action_space"],

            reward=reward,
            label=label,
            reward_cov=batch["reward_cov"],
            reward_ali=batch["reward_ali"],
            depth_raw_path=batch["depth_raw_path"],
        )
        return loss, err, dense_info
    
    def run_predict(
        self, 
        i: torch.Tensor, 
    ) -> torch.Tensor:
        """
        Args:
        - i: [B, C, H, W], float

        Return:
        - dense: [B, H, W]
        """
        o: torch.Tensor = self.net(i)
        assert len(o.shape) == 4 and o.shape[1] == 1, o.shape
        B, _, H, W = o.shape

        dense = o.squeeze(dim=1) # [B, H, W]
        assert dense.shape == (B, H, W), dense.shape
        
        return dense
    
    def log_all(self, loss: torch.Tensor, err: torch.Tensor, name: str):
        self.log_dict({f"{name}/loss": loss.detach().clone()}, sync_dist=True)
        self.log_dict({f"{name}/err": err.detach().clone()}, sync_dist=True)
    
    def log_img(self, batch_idx: int, dense_info: Dict[str, torch.Tensor]):
        all_pred = []
        all_tags = []

        all_pred.append(utils.torch_to_numpy(dense_info["dense"]))
        all_tags.append("coverage")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense"], dense_info["action_space"])))
        all_tags.append("coverage")

        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]
        action_raw = dense_info["action_raw"]
        reward = dense_info["reward"]
        label = dense_info["label"]
        reward_cov = dense_info["reward_cov"]
        reward_ali = dense_info["reward_ali"] 
        action_str = [
            f"action\n" + 
            f"i={action_raw[i, 0]:.3f} j={action_raw[i, 1]:.3f}\n" + 
            f"d={action_raw[i, 2]:.3f} a={action_raw[i, 3]:.3f}\n" + 
            f"reward\n" + 
            f"r={reward[i]:.4f} l={label[i]:.4f}\n" +
            f"c={reward_cov[i]:.4f} a={reward_ali[i]:.4f}"
            for i in range(plot_batch_size)
        ]
        depth_raw_str = []
        for i in range(plot_batch_size):
            path_full = dense_info["depth_raw_path"][i]
            path_with_endline = []
            line_width = 20
            for s in range((len(path_full) + line_width - 1) // line_width):
                path_with_endline.append(path_full[s * line_width: min((s + 1) * line_width, len(path_full))])
            depth_raw_str.append("\n".join(path_with_endline))
        learn_utils.plot_wrap(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(dense_info["mask_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth_raw"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_ij(dense_info["action_raw"])),
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_direct_ij(dense_info["action_direct_ij_raw"], "left")),
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_direct_ij(dense_info["action_direct_ij_raw"], "right")),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])],
                    widths=[3, 3, 3],
                ),
                utils.torch_to_numpy(dense_info["depth_next_raw"]),
                *all_pred,
            ],
            tag=f"dense_predict/{batch_idx}",
            titles=[depth_raw_str, "mask_raw", "depth_nmd", "mask", action_str, "depth_next"] + all_tags,
            colorbars=["gist_gray", "gist_gray", "gist_gray", "gist_gray", None, "gist_gray"] + ["viridis"] * len(all_tags),
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )
    
    def log_img_eval(
        self,
        dense_info: Dict[str, torch.Tensor],
    ) -> matplotlib.figure.Figure:
        all_pred = []
        all_tags = []

        all_pred.append(utils.torch_to_numpy(dense_info["dense"]))
        all_tags.append("coverage")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense"], dense_info["action_space"])))
        all_tags.append("coverage")
        
        action_raw = dense_info["action_raw"]
        action_str = [
            f"action\n" + 
            f"i={action_raw[i, 0]:.3f} j={action_raw[i, 1]:.3f}\n" + 
            f"d={action_raw[i, 2]:.3f} a={action_raw[i, 3]:.3f}\n"
            for i in range(action_raw.shape[0])
        ]

        return learn_utils.plot_wrap_fig(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(dense_info["mask_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth_raw"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_ij(dense_info["action_raw"])),
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_direct_ij(dense_info["action_direct_ij_raw"], "left")),
                        utils.torch_to_numpy(FunnelActEncDec.decode_fling_direct_ij(dense_info["action_direct_ij_raw"], "right")),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.]), np.array([0., 0., 1.])],
                    widths=[3, 3, 3],
                ),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(self.normalize_depth(dense_info["depth"]))[..., None], 3, axis=3),
                    actions=[utils.torch_to_numpy(dense_info["action_center_ij_rotated"])],
                    colors=[np.array([1., 0., 0.])],
                    widths=[3],
                ),
                *all_pred,
            ],
            titles=["depth_raw", "mask_raw", "depth_nmd", "mask", action_str, "action_rotated"] + all_tags,
            colorbars=["gist_gray", "gist_gray", "gist_gray", "gist_gray", None, None] + ["viridis"] * len(all_tags),
            plot_batch_size=dense_info["depth_raw"].shape[0],
        )

    def training_step(self, batch, batch_idx):
        opt:torch.optim.Optimizer = self.optimizers()
        sch:torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()

        loss, err, dense_info = self.forward_all(batch)
        self.log_all(loss, err, "train")
        
        opt.zero_grad()
        self.manual_backward(loss)
        opt.step()
        sch.step()

        if self.double_q:
            if (batch_idx + 1) % self.net_target_update_freq == 0:
                self.net_target.load_state_dict(self.net.state_dict())

    def validation_step(self, batch, batch_idx):
        loss, err, dense_info = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(loss, err, "valid")
        
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, dense_info)