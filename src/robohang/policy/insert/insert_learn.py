import os
import copy
from typing import List, Dict, Literal, Union, Tuple, Any
import json
import math
from dataclasses import dataclass
import pprint

import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch
from multiprocessing import Manager
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


REWARD_THRESHOLD = dict(left=0.95, right=0.9)

# dataset
@dataclass
class InsertDataDict:
    depth_path: str
    mask_garment_path: str
    mask_inverse_path: str
    mask_hanger_path: str
    reward: float
    action_1_ij: np.ndarray
    action_2_ij: np.ndarray


def concat_mask(mask_dict: Dict[str, np.ndarray]):
    return np.array([mask_dict["mask_garment"], mask_dict["mask_inverse"], mask_dict["mask_hanger"]])


class InsertDataset(Dataset):
    def __init__(
        self,
        data_list: List[InsertDataDict], 
        data_index_table: np.ndarray, 
        ds_cfg: omegaconf.DictConfig,
        action_1_str: str,
        action_2_str: str,
        name="none",
    ) -> None:
        super().__init__()

        self._data_list = Manager().list(data_list)
        self._data_index_table = data_index_table.copy()
        self._ds_cfg = copy.deepcopy(ds_cfg)

        self._size = int(len(self._data_index_table))
        self._dtype = getattr(np, self._ds_cfg.dtype)

        self._action_1_str = action_1_str
        self._action_2_str = action_2_str
        self._name = name

        assert (action_1_str, action_2_str) in [(x.action_1_str, x.action_2_str) for x in [InsertLeftEndModule, InsertRightEndModule]], (action_1_str, action_2_str)

        print(f"insert dataset {self._action_1_str} {self._action_2_str} {self._name}: len {self._size}")
    
    def __len__(self):
        return self._size
        
    def __getitem__(self, index: int):
        # extract data
        data: InsertDataDict = self._data_list[self._data_index_table[index]]
        depth_raw: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_garment_raw: np.ndarray = np.load(data.mask_garment_path)
        mask_inverse_raw: np.ndarray = np.load(data.mask_inverse_path)
        mask_hanger_raw: np.ndarray = np.load(data.mask_hanger_path)
        reward: float = float(data.reward)
        action_1_ij_raw = data.action_1_ij.astype(self._dtype)
        action_2_ij_raw = data.action_2_ij.astype(self._dtype)

        # augment data
        if np.random.random() < self._ds_cfg.aug.mask_inv_fail_prob:
            mask_inverse_raw = np.zeros_like(mask_inverse_raw)
        mask_raw = concat_mask(dict(mask_garment=mask_garment_raw, mask_inverse=mask_inverse_raw, mask_hanger=mask_hanger_raw))
        depth, mask, info = learn_utils.data_augmentation(depth_raw, mask_raw, self._ds_cfg.aug)

        # action space
        action_1_space = np.zeros_like(depth_raw)
        i, j = MetaInfo.get_action_space_slice(self._action_1_str, depth_raw)
        action_1_space[i, j] = 1.
        action_2_space = np.zeros_like(depth_raw)
        i, j = MetaInfo.get_action_space_slice(self._action_2_str, depth_raw)
        action_2_space[i, j] = 1.

        # random action
        H, W = depth_raw.shape
        if self._name == "train":
            weights = (np.array([.5, 1., .5])[None, :] * np.array([.5, 1., .5])[:, None]).reshape(-1)
            ijs = np.concatenate([
                np.tile(np.array([-1, 0, 1])[None, :], (3, 1))[..., None], 
                np.tile(np.array([-1, 0, 1])[:, None], (1, 3))[..., None], 
            ], axis=2).reshape(-1, 2)
            p = weights / np.sum(weights)
            action_1_ij = np.clip(action_1_ij_raw.copy() + ijs[np.random.choice(len(p), p=p)], (0, 0), (H - 1, W - 1))
            action_2_ij = np.clip(action_2_ij_raw.copy() + ijs[np.random.choice(len(p), p=p)], (0, 0), (H - 1, W - 1))
        elif self._name == "valid":
            action_1_ij = np.clip(action_1_ij_raw.copy(), (0, 0), (H - 1, W - 1))
            action_2_ij = np.clip(action_2_ij_raw.copy(), (0, 0), (H - 1, W - 1))
        else:
            raise ValueError(self._name)
        
        return dict(
            depth=depth,
            mask=mask,
            action_1_ij=action_1_ij,
            action_2_ij=action_2_ij,
            reward=reward,

            action_1_space=action_1_space,
            action_2_space=action_2_space,

            depth_raw=depth_raw,
            mask_raw=mask_raw,
            action_1_ij_raw=action_1_ij_raw,
            action_2_ij_raw=action_2_ij_raw,

            depth_raw_path=os.path.relpath(data.depth_path, utils.get_path_handler()(".")),
        )


def make_insert_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    make_cfg: omegaconf.DictConfig,
    ds_cfg: omegaconf.DictConfig,
    endpoint_name: Literal["left", "right"]
):
    pattern = "^completed.txt$"
    df = learn_utils.DataForest(data_path, [pattern])
    node_n_raw = df.get_forest_size(pattern)
    
    data_list = []

    positive_cnt = 0
    negative_cnt = 0

    name_str = dict(
        action_1=0,
        action_2=1,
        error_before=0,
        error_after=2,
        state_before=0,
        obs=0,
    )
    reward_file_left = "2.json"
    reward_file_right = "4.json"
    if endpoint_name == "left":
        name_str = omegaconf.DictConfig(name_str)
    elif endpoint_name == "right":
        name_str = omegaconf.DictConfig({
            k: v + 2 # (0 if k == "reward" else 2) 
        for k, v in name_str.items()})
    else:
        raise ValueError(endpoint_name)
    
    print("scanning all trajectories ...")
    for idx in tqdm.tqdm(range(node_n_raw)):
        misc_path = os.path.join(os.path.dirname(df.get_item(pattern, idx).file_path), "misc.json")
        with open(misc_path, "r") as f_obj:
            misc_info = json.load(f_obj)
        base_dir = os.path.dirname(misc_path)

        def format_batch(batch_idx):
            return str(batch_idx).zfill(len(str(misc_info["batch_size"] - 1)))

        action_1_xy_batch = np.load(os.path.join(base_dir, "action", f"{name_str.action_1}.npy"), allow_pickle=True).item()["xy"]
        action_2_xy_batch = np.load(os.path.join(base_dir, "action", f"{name_str.action_2}.npy"), allow_pickle=True).item()["xy"]
        reward_batch = dict()
        with open(os.path.join(base_dir, "score", reward_file_left), "r") as f_obj:
            reward_batch["left"] = json.load(f_obj)
        with open(os.path.join(base_dir, "score", reward_file_right), "r") as f_obj:
            reward_batch["right"] = json.load(f_obj)
        with open(os.path.join(base_dir, "sim_error", f"{name_str.error_before}.json"), "r") as f_obj:
            sim_error_before_batch = json.load(f_obj)
        with open(os.path.join(base_dir, "sim_error", f"{name_str.error_after}.json"), "r") as f_obj:
            sim_error_after_batch = json.load(f_obj)
        # state_before_batch = np.load(os.path.join(base_dir, "state", f"{name_str.state_before}.npy"), allow_pickle=True).item()

        for batch_idx in range(int(misc_info["batch_size"])):
            if sim_error_before_batch["all"][batch_idx]: # skip sim_error batch
                continue

            if endpoint_name == "right" and reward_batch["left"]["left"][batch_idx] < REWARD_THRESHOLD["left"]: # when insert right point, skip if left is not inserted
                continue

            # obs
            obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))
            depth_path = os.path.join(obs_base_dir, "reproject_depth", f"{name_str.obs}.npy")
            if bool(make_cfg.mask_interp):
                mask_garment_path = os.path.join(obs_base_dir, "reproject_is_garment", f"{name_str.obs}.npy")
                mask_inverse_path = os.path.join(obs_base_dir, "reproject_is_inverse", f"{name_str.obs}.npy")
                mask_hanger_path = os.path.join(obs_base_dir, "reproject_is_hanger", f"{name_str.obs}.npy")
            else:
                mask_garment_path = os.path.join(obs_base_dir, "reproject_is_garment_nointerp", f"{name_str.obs}.npy")
                mask_inverse_path = os.path.join(obs_base_dir, "reproject_is_inverse_nointerp", f"{name_str.obs}.npy")
                mask_hanger_path = os.path.join(obs_base_dir, "reproject_is_hanger_nointerp", f"{name_str.obs}.npy")

            with open(os.path.join(obs_base_dir, "info", f"{name_str.obs}.json"), "r") as f_obj:
                obs_info = json.load(f_obj)
            assert MetaInfo.check_reproject_info(obs_info["reproject"]), str(obs_info["reproject"])

            # reward
            reward = (
                (reward_batch[endpoint_name][endpoint_name][batch_idx] >= REWARD_THRESHOLD[endpoint_name]) and
                (not sim_error_after_batch["all"][batch_idx])
            )
            if reward:
                positive_cnt += 1
            else:
                negative_cnt += 1
            reward = float(reward)

            # action
            depth_img = np.load(depth_path)
            z = MetaInfo.calculate_z_world(depth_img)
            action_1_ij = policy_utils.xyz2ij(
                action_1_xy_batch[batch_idx, 0],
                action_1_xy_batch[batch_idx, 1],
                z, MetaInfo.reproject_camera_info
            )
            action_2_ij = policy_utils.xyz2ij(
                action_2_xy_batch[batch_idx, 0],
                action_2_xy_batch[batch_idx, 1],
                z, MetaInfo.reproject_camera_info
            )

            # assemble data
            data_dict = InsertDataDict(
                depth_path=depth_path,
                mask_garment_path=mask_garment_path,
                mask_inverse_path=mask_inverse_path,
                mask_hanger_path=mask_hanger_path,
                reward=reward,
                action_1_ij=action_1_ij,
                action_2_ij=action_2_ij,
            )
            data_list.append(data_dict)
    
    print(f"pos:{positive_cnt} neg:{negative_cnt}")
    total_size = len(data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    if endpoint_name == "left":
        action_1_str = "press"
        action_2_str = "lift"
    elif endpoint_name == "right":
        action_1_str = "drag"
        action_2_str = "rotate"
    else:
        raise ValueError(endpoint_name)
    trds = InsertDataset(data_list, path_idx_permutated[valid_size:], ds_cfg, action_1_str, action_2_str, name=f"train")
    vlds = InsertDataset(data_list, path_idx_permutated[:valid_size], ds_cfg, action_1_str, action_2_str, name=f"valid")
    return trds, vlds


class DualArmModule(pl.LightningModule):
    action_1_str: str
    action_2_str: str
    action_1_color: np.ndarray
    action_2_color: np.ndarray
    action_1_color_name: str
    action_2_color_name: str

    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.nets: Dict[Literal["action_1", "action_2"], torch.nn.Module] = torch.nn.ModuleDict()

        kwargs = copy.deepcopy(model_kwargs["net1"])
        kwargs["input_channel"] = len(model_kwargs["mask_dims"]) + 1
        kwargs["output_channel"] = 1
        self.nets["action_1"] = UNet(**kwargs)

        kwargs = copy.deepcopy(model_kwargs["net2"])
        kwargs["input_channel"] = len(model_kwargs["mask_dims"]) + 2
        kwargs["output_channel"] = 1
        self.nets["action_2"] = UNet(**kwargs)

        self.d_unit = float(model_kwargs["d_unit"])
        self.mask_dims = [int(_) for _ in model_kwargs["mask_dims"]]
        self.net1_start_step = int(learn_kwargs["net1_start_step"])
        self.net1_use_binary_lable = bool(learn_kwargs["net1_use_binary_lable"])
        self.positive_weight = float(learn_kwargs["positive_weight"])
        self.negative_weight = float(learn_kwargs["negative_weight"])
        self.automatic_optimization = False

    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"][a]["name"])(
                self.nets[a].parameters(), 
                **(self.learn_kwargs["optimizer"][a]["cfg"]),
            ) for i, a in enumerate(["action_1", "action_2"])
        ]
        schedule = [
            getattr(torch.optim.lr_scheduler, self.learn_kwargs["schedule"][a]["name"])(
                optimizer[i],
                **(self.learn_kwargs["schedule"][a]["cfg"]),
            ) for i, a in enumerate(["action_1", "action_2"])
        ]
        return optimizer, schedule
    
    def normalize_depth(self, depth: torch.Tensor):
        B = depth.shape[0]
        return (depth - depth.view(B, -1).mean(-1).view(B, 1, 1)) / self.d_unit
    
    def _preprocess_input(self, depth: torch.Tensor, mask: torch.Tensor):
        """
        `d = (d - mean) / unit`

        Args:
        - depth: [B, H, W]
        - mask: [B, 3, H, W]

        Return:
        - i: [B, 4, H, W]
        """
        B, H, W = depth.shape
        depth = self.normalize_depth(depth)
        return torch.concat([depth.unsqueeze(1), mask], dim=1)
    
    def preprocess_input_1(self, depth: torch.Tensor, mask: torch.Tensor):
        """
        Args:
        - depth: [B, H, W]
        - mask: [B, 3, H, W]

        Return:
        - i: [B, 4, H, W]
        """
        return self._preprocess_input(depth, mask)[:, [0] + [x + 1 for x in self.mask_dims], :, :]
    
    def preprocess_input_2(self, depth: torch.Tensor, mask: torch.Tensor, action_1_ij: torch.Tensor):
        """
        Args:
        - depth: [B, H, W]
        - mask: [B, 3, H, W]
        - action_1_ij: [B, 2]

        Return:
        - i: [B, C, H, W]
        """
        return torch.concat([
            learn_utils.one_hot(action_1_ij, depth.shape, depth.dtype, depth.device).unsqueeze(dim=1),
            self._preprocess_input(depth, mask),
        ], dim=1)[:, [0, 1] + [x + 2 for x in self.mask_dims], :, :]

    def extract_data(self, batch):
        """
        Return:
        - depth: [B, H, W]
        - mask: [B, 2, H, W]
        - action_1_ij: [B, 2], long
        - action_2_ij: [B, 2], long
        - reward: [B, ]

        - action_1_space: [B, H, W]
        - action_2_space: [B, H, W]
        """
        depth: torch.Tensor = batch["depth"] # [B, H, W]
        mask: torch.Tensor = batch["mask"] # [B, H, W]
        action_1_ij: torch.Tensor = batch["action_1_ij"] # [B, 2]
        action_2_ij: torch.Tensor = batch["action_2_ij"] # [B, 2]
        reward: torch.Tensor = batch["reward"] # [B, ]

        action_1_space: torch.Tensor = batch["action_1_space"] # [B, H, W]
        action_2_space: torch.Tensor = batch["action_2_space"] # [B, H, W]

        B, H, W = depth.shape
        assert mask.shape == (B, 3, H, W), mask.shape
        assert action_1_ij.shape == (B, 2), action_1_ij.shape
        assert action_2_ij.shape == (B, 2), action_2_ij.shape
        assert reward.shape == (B, ), reward.shape

        assert action_1_space.shape == depth.shape, action_1_space.shape
        assert action_2_space.shape == depth.shape, action_2_space.shape

        return depth, mask, action_1_ij.to(torch.long), action_2_ij.to(torch.long), reward, action_1_space, action_2_space
    
    def run_forward(
        self, 
        i_1: torch.Tensor, 
        i_2: torch.Tensor, 
        ij_1: torch.Tensor, 
        ij_2: torch.Tensor, 
        as_2: torch.Tensor,
    ):
        """
        Args:
        - i_1: [B, Cs, H, W], float
        - i_2: [B, Ce, H, W], float
        - ij_1: [B, 2], int
        - ij_2: [B, 2], int
        - as_2: [B, H, W], float

        Return:
        - pred_1: [B]
        - dense_1: [B, H, W]
        - pred_2: [B]
        - dense_2: [B, H, W]
        - label_1: [B]
        """
        device = self.device
        
        o_1: torch.Tensor = self.nets["action_1"](i_1)
        assert len(o_1.shape) == 4 and o_1.shape[1] == 1, o_1.shape
        o_2: torch.Tensor = self.nets["action_2"](i_2)
        assert len(o_2.shape) == 4 and o_2.shape[1] == 1, o_2.shape

        dense_1 = o_1.squeeze(dim=1) # [B, H, W]
        dense_2 = o_2.squeeze(dim=1) # [B, H, W]
        B, H, W = dense_1.shape

        pred_1 = dense_1[torch.arange(B).to(device), ij_1[:, 0], ij_1[:, 1]]
        pred_2 = dense_2[torch.arange(B).to(device), ij_2[:, 0], ij_2[:, 1]]
        if self.net1_use_binary_lable:
            label_1: torch.Tensor = (learn_utils.out_of_action_space_to_min(dense_2, as_2).view(B, -1).max(dim=-1).values > 0.).float()
        else:
            label_1 = torch.nn.functional.sigmoid(learn_utils.out_of_action_space_to_min(dense_2, as_2).view(B, -1).max(dim=-1).values).detach().clone()

        assert pred_1.shape == (B, ), pred_1.shape
        assert dense_1.shape == (B, H, W), dense_1.shape
        assert pred_2.shape == (B, ), pred_2.shape
        assert dense_2.shape == (B, H, W), dense_2.shape
        assert label_1.shape == (B, ), label_1.shape
        
        return [pred_1, dense_1, pred_2, dense_2, label_1]
    
    def get_weight(self, gt_binary: torch.Tensor):
        assert (0. <= gt_binary).all() and (gt_binary <= 1.).all(), gt_binary
        return self.negative_weight + (self.positive_weight - self.negative_weight) * gt_binary
    
    def forward_all(self, batch):
        depth, mask, action_1_ij, action_2_ij, reward, action_1_space, action_2_space = self.extract_data(batch)

        i_1 = self.preprocess_input_1(depth, mask)
        i_2 = self.preprocess_input_2(depth, mask, action_1_ij)
        pred_1, dense_1, pred_2, dense_2, label_1 = self.run_forward(i_1=i_1, i_2=i_2, ij_1=action_1_ij, ij_2=action_2_ij, as_2=action_2_space)

        loss_1 = torch.nn.functional.binary_cross_entropy_with_logits(pred_1, label_1, self.get_weight(label_1))
        err_1 = torch.logical_xor(pred_1 > 0., label_1 >= 0.5).float().mean()
        loss_2 = torch.nn.functional.binary_cross_entropy_with_logits(pred_2, reward, self.get_weight(reward))
        err_2 = torch.logical_xor(pred_2 > 0., reward >= 0.5).float().mean()
        
        dense_info = dict(
            depth=depth,
            mask=mask,
            dense_1=dense_1,
            dense_2=dense_2,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],
            action_1_space=action_1_space,
            action_2_space=action_2_space,
            action_1_ij_raw=batch["action_1_ij_raw"],
            action_2_ij_raw=batch["action_2_ij_raw"],
            label_1=label_1,
            reward=reward,

            depth_raw_path=batch["depth_raw_path"],
        )
        return loss_1, loss_2, err_1, err_2, dense_info
    
    def get_loss_1_weight(self):
        return 0.0 if self.global_step < self.net1_start_step else 1.0
    
    def training_step(self, batch, batch_idx):
        opt1, opt2 = self.optimizers()
        sch1, sch2 = self.lr_schedulers()

        loss_1, loss_2, err_1, err_2, dense_info = self.forward_all(batch)
        self.log_all(loss_1, loss_2, err_1, err_2, "train")
        
        opt1.zero_grad()
        self.manual_backward(loss_1 * self.get_loss_1_weight())
        opt1.step()
        sch1.step()

        opt2.zero_grad()
        self.manual_backward(loss_2)
        opt2.step()
        sch2.step()

    def validation_step(self, batch, batch_idx):
        loss_1, loss_2, err_1, err_2, dense_info = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(loss_1, loss_2, err_1, err_2, "valid")
        
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, dense_info)
    
    def log_all(self, loss_1: torch.Tensor, loss_2: torch.Tensor, err_1: torch.Tensor, err_2: torch.Tensor, name: str):
        self.log(f"{name}/loss1", loss_1.detach().cpu())
        self.log(f"{name}/loss2", loss_2.detach().cpu())
        self.log(f"{name}/err1", err_1.detach().cpu())
        self.log(f"{name}/err2", err_2.detach().cpu())
        self.log(f"loss_1_weight", self.get_loss_1_weight())
        self.log(f"positive_weight", self.positive_weight)
        self.log(f"negative_weight", self.negative_weight)
    
    def log_img(self, batch_idx: int, dense_info: Dict[str, torch.Tensor]):
        all_pred = []
        all_tags = []

        all_pred.append(utils.torch_to_numpy(dense_info["dense_1"]))
        all_tags.append(f"action_1")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense_1"], dense_info["action_1_space"])))
        all_tags.append(f"action_1_masked")
        all_pred.append(utils.torch_to_numpy(dense_info["dense_2"]))
        all_tags.append(f"action_2")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense_2"], dense_info["action_2_space"])))
        all_tags.append(f"action_2_masked")
        
        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]
        action_label_list = []
        depth_raw_str = []
        for batch_idx in range(plot_batch_size):
            action_label_list.append(
                f"{self.action_1_color_name}:{self.action_1_str} {self.action_2_color_name}:{self.action_2_str}\n"
                f"label_1:{dense_info['label_1'][batch_idx]:.4f} "
                f"reward:{dense_info['reward'][batch_idx]}"
            )
            path_full = dense_info["depth_raw_path"][batch_idx]
            path_with_endline = []
            line_width = 20
            for s in range((len(path_full) + line_width - 1) // line_width):
                path_with_endline.append(path_full[s * line_width: min((s + 1) * line_width, len(path_full))])
            depth_raw_str.append("\n".join(path_with_endline))

        learn_utils.plot_wrap(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"][:, 0, :, :]),
                utils.torch_to_numpy(dense_info["mask"][:, 1, :, :]),
                utils.torch_to_numpy(dense_info["mask"][:, 2, :, :]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth_raw"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["action_1_ij_raw"]).astype(np.int32),
                        utils.torch_to_numpy(dense_info["action_2_ij_raw"]).astype(np.int32),
                    ],
                    colors=[self.action_1_color, self.action_2_color],
                    widths=[3, 3],
                ),
                *all_pred,
            ],
            tag=f"dense_predict_{self.action_1_str}_{self.action_2_str}",
            titles=[depth_raw_str, "depth_nmd", "mask_garment", "mask_inverse", "mask_hanger", action_label_list] + all_tags,
            colorbars=["gist_gray"] * 5 + [None] + ["viridis"] * len(all_tags),
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )
    
    def run_predict(
        self,
        depth: torch.Tensor,
        mask: torch.Tensor,
        action_1_space: torch.Tensor,
        action_2_space: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
        - depth: [B, H, W]
        - m: [B, 3, H, W]
        - action_1_space: [B, H, W]
        - action_2_space: [B, H, W]

        Return: 
        - action_1_xy: [B, 2]
        - action_2_xy: [B, 2]
        - dense_info: str -> Tensor
        """
        B, H, W = depth.shape
        assert mask.shape == (B, 3, H, W), mask.shape
        assert action_1_space.shape == (B, H, W), action_1_space.shape
        assert action_2_space.shape == (B, H, W), action_2_space.shape

        # action 1
        i_1 = self.preprocess_input_1(depth, mask)
        o_1: torch.Tensor = self.nets["action_1"](i_1)
        assert len(o_1.shape) == 4 and o_1.shape[1] == 1, o_1.shape
        dense_1 = o_1.squeeze(dim=1) # [B, H, W]

        dense_1_masked = learn_utils.out_of_action_space_to_min(dense_1, action_1_space)
        best_indices = dense_1_masked.view(B, -1).max(dim=-1).indices # [B, ]
        I_idx = best_indices // W # [B, ]
        J_idx = best_indices % W # [B, ]
        action_1_ij = torch.concat([I_idx[:, None], J_idx[:, None]], dim=1) # [B, 2]
        action_1_xy = policy_utils.action_batch_ij_to_xy(action_1_ij, depth)

        # action 2
        i_2 = self.preprocess_input_2(depth, mask, action_1_ij)
        o_2: torch.Tensor = self.nets["action_2"](i_2)
        assert len(o_2.shape) == 4 and o_2.shape[1] == 1, o_2.shape
        dense_2 = o_2.squeeze(dim=1) # [B, H, W]

        dense_2_masked = learn_utils.out_of_action_space_to_min(dense_2, action_2_space)
        best_indices = dense_2_masked.view(B, -1).max(dim=-1).indices # [B, ]
        I_idx = best_indices // W # [B, ]
        J_idx = best_indices % W # [B, ]
        action_2_ij = torch.concat([I_idx[:, None], J_idx[:, None]], dim=1) # [B, 2]
        action_2_xy = policy_utils.action_batch_ij_to_xy(action_2_ij, depth)
        
        # info
        dense_info = dict(
            depth=depth,
            mask=mask,
            action_1_space=action_1_space,
            action_2_space=action_2_space,

            dense_1=dense_1,
            dense_2=dense_2,
            action_1_ij=action_1_ij,
            action_2_ij=action_2_ij,
        )
        return action_1_xy, action_2_xy, dense_info
    
    def log_img_eval(
        self,
        dense_info: Dict[str, torch.Tensor],
    ) -> matplotlib.figure.Figure:
        all_pred = []
        all_tags = []

        all_pred.append(utils.torch_to_numpy(dense_info["dense_1"]))
        all_tags.append(f"action_1")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense_1"], dense_info["action_1_space"])))
        all_tags.append(f"action_1_masked")
        all_pred.append(utils.torch_to_numpy(dense_info["dense_2"]))
        all_tags.append(f"action_2")
        all_pred.append(utils.torch_to_numpy(learn_utils.out_of_action_space_to_min(dense_info["dense_2"], dense_info["action_2_space"])))
        all_tags.append(f"action_2_masked")
        action_str = f"{self.action_1_color_name}:{self.action_1_str} {self.action_2_color_name}:{self.action_2_str}"

        return learn_utils.plot_wrap_fig(
            denses=[
                utils.torch_to_numpy(dense_info["depth"]),
                utils.torch_to_numpy(dense_info["mask"][:, 0, :, :]),
                utils.torch_to_numpy(dense_info["mask"][:, 1, :, :]),
                utils.torch_to_numpy(dense_info["mask"][:, 2, :, :]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["action_1_ij"]).astype(np.int32),
                        utils.torch_to_numpy(dense_info["action_2_ij"]).astype(np.int32),
                    ],
                    colors=[self.action_1_color, self.action_2_color],
                    widths=[3, 3],
                ),
                *all_pred,
            ],
            titles=["depth", "mask_garment", "mask_inverse", "mask_hanger", "depth_nmd", action_str] + all_tags,
            colorbars=["gist_gray"] * 5 + [None] + ["viridis"] * len(all_tags),
            plot_batch_size=dense_info["depth"].shape[0],
        )


class InsertLeftEndModule(DualArmModule):
    action_1_str = "press"
    action_2_str = "lift"
    action_1_color = np.array([1., 0., 0.])
    action_2_color = np.array([0., 1., 0.])
    action_1_color_name = "red"
    action_2_color_name = "green"


class InsertRightEndModule(DualArmModule):
    action_1_str = "drag"
    action_2_str = "rotate"
    action_1_color = np.array([0., 0., 1.])
    action_2_color = np.array([1., 1., 0.])
    action_1_color_name = "blue"
    action_2_color_name = "yellow"