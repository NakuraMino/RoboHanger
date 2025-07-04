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

import trimesh.transformations as tra
import trimesh
import omegaconf

import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
import robohang.policy.policy_utils as policy_utils
from robohang.policy.net import CNN
import robohang.common.utils as utils
from robohang.policy.funnel.funnel_learn import rotate_and_translate

import lightning.pytorch as pl


@dataclass
class KeypointsDataDict:
    depth_path: str
    mask_path: str
    ij_left: np.ndarray
    ij_right: np.ndarray
    faceup: bool


class KeypointsDataset(Dataset):
    def __init__(
        self,
        data_list:List[KeypointsDataDict],
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

        if name == "train":
            pass
        elif name == "valid":
            pass
        else:
            raise ValueError(name)
        self._name = name

        print(f"funnel dataset {name}: len {self._size}")
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index: int):
        # extract data
        data: KeypointsDataDict = self._data_list[self._data_index_table[index]]
        depth_raw: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_raw: np.ndarray = np.load(data.mask_path).astype(self._dtype)
        
        ij_left_raw: np.ndarray = data.ij_left.copy().astype(self._dtype)
        ij_right_raw: np.ndarray = data.ij_right.copy().astype(self._dtype)
        faceup: bool = data.faceup

        # augment data
        depth, mask, info = learn_utils.data_augmentation(depth_raw, mask_raw, self._ds_cfg.aug)
        if self._name == "train":
            depth, mask, ij_left, ij_right = rotate_and_translate(self._ds_cfg.aug, depth, mask, ij_left_raw, ij_right_raw)
        elif self._name == "valid":
            ij_left = ij_left_raw.copy()
            ij_right = ij_right_raw.copy()
        else:
            raise ValueError(self._name)
        
        return dict(
            depth=depth,
            mask=mask,
            mask_unflip=info["mask_unflip"],
            ij_left=ij_left,
            ij_right=ij_right,
            faceup=faceup,
            
            depth_raw=depth_raw,
            mask_raw=mask_raw,
            ij_left_raw=ij_left_raw,
            ij_right_raw=ij_right_raw,
            depth_path=data.depth_path,
        )


def make_keypoint_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    make_cfg: omegaconf.DictConfig,
    ds_cfg: omegaconf.DictConfig,
):
    pattern = "^completed.txt$"
    df = learn_utils.DataForest(data_path, [pattern])
    node_n_raw = df.get_forest_size(pattern)

    all_data_list: List[KeypointsDataDict] = []
    
    print("scanning all trajectories ...")
    for idx in tqdm.tqdm(range(node_n_raw)):
        misc_path = os.path.join(os.path.dirname(df.get_item(pattern, idx).file_path), "misc.json")
        with open(misc_path, "r") as f_obj:
            misc_info: dict = json.load(f_obj)
        base_dir = os.path.dirname(misc_path)
        keypoints_dict = misc_info["garment_keypoints"]

        def format_step(step_idx):
            return str(step_idx).zfill(len(str(2 * misc_info["num_trial"])))
        
        def format_batch(batch_idx):
            return str(batch_idx).zfill(len(str(misc_info["batch_size"] - 1)))
        
        for depth_idx in range(int(misc_info["num_trial"]) * 2 + 1):
            if make_cfg.skip_pick_place and depth_idx % 2 == 1: # skip pick place
                continue
            with open(os.path.join(base_dir, "sim_error", format_step(depth_idx) + ".json"), "r") as f_obj:
                sim_error_batch = json.load(f_obj)
            with open(os.path.join(base_dir, "score", format_step(depth_idx) + ".json"), "r") as f_obj:
                score_batch = json.load(f_obj)
            state = np.load(os.path.join(base_dir, "state", format_step(depth_idx) + ".npy"), allow_pickle=True).item()
            for batch_idx in range(int(misc_info["batch_size"])):
                if sim_error_batch["all"][batch_idx]: # skip sim_error batch
                    continue
                if score_batch["coverage"][batch_idx] < make_cfg.coverage_threshold: # use data with large coverage
                    continue

                # obs
                obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))
                depth_path = os.path.join(obs_base_dir, "reproject_depth", format_step(depth_idx) + ".npy")
                mask_path = os.path.join(obs_base_dir, "reproject_is_garment", format_step(depth_idx) + ".npy")
                with open(os.path.join(obs_base_dir, "info", format_step(depth_idx) + ".json"), "r") as f_obj:
                    obs_info = json.load(f_obj)
                assert MetaInfo.check_reproject_info(obs_info["reproject"]), str(obs_info["reproject"])

                # gt pixel
                z = MetaInfo.calculate_z_world(np.load(depth_path))
                vert = state["sim_env"]["sim"]["cloth"]["garment"]["pos"][batch_idx, ...]
                faceup = bool(score_batch["orientation"]["flip_y"][batch_idx] == 0)

                if faceup or (not make_cfg.exchange_left_right):
                    left_str, right_str = "upper_left", "upper_right"
                else:
                    left_str, right_str = "upper_right", "upper_left" # exchange left and right for better labeling
                ij_left = policy_utils.xyz2ij(
                    vert[keypoints_dict[left_str], 0],
                    vert[keypoints_dict[left_str], 1],
                    z, MetaInfo.reproject_camera_info
                )
                ij_right = policy_utils.xyz2ij(
                    vert[keypoints_dict[right_str], 0],
                    vert[keypoints_dict[right_str], 1],
                    z, MetaInfo.reproject_camera_info
                )

                # assemble data
                all_data_list.append(KeypointsDataDict(
                    depth_path=depth_path, 
                    mask_path=mask_path,
                    ij_left=ij_left,
                    ij_right=ij_right,
                    faceup=faceup,
                ))

    total_size = len(all_data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    trds = KeypointsDataset(all_data_list, path_idx_permutated[valid_size:], ds_cfg, name=f"train")
    vlds = KeypointsDataset(all_data_list, path_idx_permutated[:valid_size], ds_cfg, name=f"valid")
    return trds, vlds


class KeypointsModule(pl.LightningModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)
        self.out_of_mask_penalty = 10.
        self.penalty_pixel = self.learn_kwargs["penalty_pixel"]

        self.nets = torch.nn.ModuleDict({})
        self.pred_target_str = copy.deepcopy(model_kwargs["pred_target_str"])
        for k in self.pred_target_str:
            kwargs = copy.deepcopy(model_kwargs["net"])
            kwargs["input_channel"] = 2
            kwargs["input_height"] = MetaInfo.reproject_height()
            kwargs["input_width"] = MetaInfo.reproject_width()
            if k in ["left", "right"]:
                kwargs["output_channel"] = 3
            elif k == "faceup":
                kwargs["output_channel"] = 1
            else:
                raise ValueError(k)
            self.nets[k] = CNN(**kwargs)
            
        self.d_unit = float(model_kwargs["d_unit"])
        self.lr_loss_func = getattr(torch.nn.functional, self.learn_kwargs["loss_lr"])

        self.automatic_optimization = False
        
    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
                self.nets.parameters(), 
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
        - i: [B, C, H, W]
        """
        B, H, W = depth.shape
        depth = self.normalize_depth(depth)
        return torch.concat([depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)
    
    def extract_data(self, batch):
        """
        Return:
        - depth: [B, H, W]
        - mask: [B, H, W]
        - mask_unflip: [B, H, W]
        - action: [B, 4]
        - reward: dict, str -> [B]
        """
        depth: torch.Tensor = batch["depth"] # [B, H, W]
        mask: torch.Tensor = batch["mask"] # [B, H, W]
        mask_unflip: torch.Tensor = batch["mask_unflip"] # [B, H, W]
        ij_left: torch.Tensor = batch["ij_left"] # [B, 2]
        ij_right: torch.Tensor = batch["ij_right"] # [B, 2]
        faceup: torch.Tensor = batch["faceup"] # [B], bool

        B, H, W = depth.shape
        assert mask.shape == depth.shape, mask.shape
        assert mask_unflip.shape == depth.shape, mask_unflip.shape
        assert ij_left.shape == (B, 2), ij_left.shape
        assert ij_right.shape == (B, 2), ij_right.shape
        assert faceup.shape == (B, ), faceup.shape

        return depth, mask, mask_unflip, ij_left, ij_right, faceup
    
    def run_predict(self, i: torch.Tensor):
        pred: Dict[str, torch.Tensor] = {}
        B, C, H, W = i.shape
        for k in self.pred_target_str:
            o: torch.Tensor = self.nets[k](i)
            if k in ["left", "right"]:
                assert len(o.shape) == 2 and o.shape[1] == 3, o.shape
                o = torch.concat([
                    (o[:, [0]] + 0.5) * H, (o[:, [1]] + 0.5) * W, 
                    torch.nn.functional.elu(o[:, [2]]) + 1.
                ], dim=1)
            else:
                assert len(o.shape) == 2 and o.shape[1] == 1, o.shape
                o = o.view(o.shape[0])
            pred[k] = o
        return pred
    
    def is_out_of_mask_weight(self, mask: torch.Tensor, ij: torch.Tensor):
        mask_threshold = 0.8
        (B, H, W), device = mask.shape, mask.device
        need_penalty = torch.logical_or(
            mask[
                torch.arange(0, B, device=device, dtype=torch.long), 
                torch.clamp(ij[:, 0].to(dtype=torch.long), 0, H - 1), 
                torch.clamp(ij[:, 1].to(dtype=torch.long), 0, W - 1),
            ] < 0.8,
            torch.logical_or(
                torch.logical_or(ij[:, 0] < 0., ij[:, 0] >= H),
                torch.logical_or(ij[:, 1] < 0., ij[:, 1] >= W),
            )
        )
        for di in range(-self.penalty_pixel, +self.penalty_pixel + 1):
            for dj in range(-self.penalty_pixel, +self.penalty_pixel + 1):
                need_penalty = torch.logical_or(mask[
                    torch.arange(0, B, device=device, dtype=torch.long), 
                    torch.clamp(ij[:, 0].to(dtype=torch.long) + di, 0, H - 1), 
                    torch.clamp(ij[:, 1].to(dtype=torch.long) + dj, 0, W - 1),
                ] < 0.8, need_penalty)
        weight = torch.where(need_penalty, self.out_of_mask_penalty, 1.)
        return weight
    
    def forward_all(self, batch):
        depth, mask, mask_unflip, ij_left, ij_right, faceup = self.extract_data(batch)
        i = self.preprocess_input(depth, mask)
        ij_lr = dict(left=ij_left, right=ij_right)

        pred = self.run_predict(i)
        
        loss_dict = {}
        err_dict = {}
        loss_total = None
        for k in self.pred_target_str:
            if k in ["left", "right"]:
                weight = self.is_out_of_mask_weight(mask_unflip, pred[k][:, :2])
                l2err = (pred[k][:, :2] - ij_lr[k]).norm(dim=1)

                lw = (self.lr_loss_func(pred[k][:, :2], ij_lr[k], reduction="none").sum(dim=1) * weight).mean() # scalar
                loss_dict[f"{k}_w"] = lw
                le = torch.abs(l2err.detach() - pred[k][:, 2]).mean()
                err_dict[f"{k}_p"] = le
                loss_total = ((loss_total + lw + le) if (loss_total is not None) else lw + le)

                err_dict[f"{k}_w"] = (l2err * weight).mean() # weighted err
                err_dict[f"{k}_e"] = l2err.mean() # unweighted err
            elif k == "faceup":
                loss_dict[k] = torch.nn.functional.binary_cross_entropy_with_logits(pred[k], faceup.float()) # scalar
                err_dict[k] = ((pred[k] > 0.) != faceup).float().mean() # scalar
                loss_total = ((loss_total + loss_dict[k]) if (loss_total is not None) else loss_dict[k])
            else:
                raise ValueError(k)
                
        
        dense_info = dict(
            pred=pred,
            depth=depth,
            mask=mask,
            ij_left=ij_left,
            ij_right=ij_right,
            faceup=faceup,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],
            ij_left_raw=batch["ij_left_raw"],
            ij_right_raw=batch["ij_right_raw"],
            depth_path=batch["depth_path"],
        )
        return loss_dict, err_dict, loss_total, dense_info
    
    def log_all(self, loss_dict: Dict[str, torch.Tensor], err_dict: Dict[str, torch.Tensor], loss_total: torch.Tensor, name: str):
        for k in err_dict.keys():
            self.log(f"{name}/{k}_err", err_dict[k].detach().cpu())
        for k in loss_dict.keys():
            self.log(f"{name}/{k}_loss", loss_dict[k].detach().cpu())
        self.log_dict({f"{name}/total_loss": loss_total.detach().cpu()})
    
    def log_img(self, batch_idx, dense_info):
        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]
        act_str_raw = []
        act_str_aug = []
        pred_str = []
        for i in range(plot_batch_size):
            path_full = dense_info["depth_path"][i]
            path_with_endline = []
            line_width = 30
            for s in range((len(path_full) + line_width - 1) // line_width):
                path_with_endline.append(path_full[s * line_width: min((s + 1) * line_width, len(path_full))])
            act_str_raw.append(
                f"gt raw red=left green=right\n" + 
                f"faceup={int(dense_info['faceup'][i])} pred={(dense_info['pred']['faceup'][i] if 'faceup' in dense_info['pred'].keys() else 0.5):.5f}\n"
            )
            act_str_aug.append(
                "gt red=left green=right\n" + 
                "\n".join(path_with_endline)
            )
            pred_str.append(
                f"pred\nleft_err={dense_info['pred']['left'][i, 2]:.4f} right_err={dense_info['pred']['right'][i, 2]:.4f}"
            )

        learn_utils.plot_wrap(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(dense_info["mask_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth_raw"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["ij_left_raw"].to(torch.long)),
                        utils.torch_to_numpy(dense_info["ij_right_raw"].to(torch.long)),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                    widths=[3, 3],
                ),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["ij_left"].to(torch.long)),
                        utils.torch_to_numpy(dense_info["ij_right"].to(torch.long)),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                    widths=[3, 3],
                ),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["pred"]["left"][:, :2].to(torch.long)),
                        utils.torch_to_numpy(dense_info["pred"]["right"][:, :2].to(torch.long)),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                    widths=[3, 3],
                ),
            ],
            tag=f"dense_predict/{batch_idx}",
            titles=["depth_raw", "mask_raw", "depth_nmd", "mask", act_str_raw, act_str_aug, pred_str],
            colorbars=["gist_gray", "gist_gray", "gist_gray", "gist_gray", None, None, None],
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )
    
    def log_img_eval(self, depth: torch.Tensor, mask: torch.Tensor, pred: Dict[str, torch.Tensor]):
        B = depth.shape[0]
        pred_str = []
        for i in range(B):
            pred_str.append(
                f"keypoints red=left, green=right\n"
                f"left_err={pred['left'][i, 2]:.4f} "
                f"right_err={pred['right'][i, 2]:.4f}"
            )
        
        return learn_utils.plot_wrap_fig(
            [learn_utils.annotate_action(
                denseRGB=np.repeat(utils.torch_to_numpy(depth)[..., None], 3, axis=3),
                actions=[
                    utils.torch_to_numpy(pred["left"][:, :2]).astype(np.int32),
                    utils.torch_to_numpy(pred["right"][:, :2]).astype(np.int32),
                ],
                colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                widths=[3, 3],
            ), utils.torch_to_numpy(mask),
            ],
            [pred_str, "mask"],
            ["gist_gray", "gist_gray"],
            1,
        )

    def training_step(self, batch, batch_idx):
        opt:torch.optim.Optimizer = self.optimizers()
        sch:torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()

        loss_dict, err_dict, loss_total, dense_info = self.forward_all(batch)
        self.log_all(loss_dict, err_dict, loss_total, "train")
        
        opt.zero_grad()
        self.manual_backward(loss_total)
        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx):
        loss_dict, err_dict, loss_total, dense_info = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(loss_dict, err_dict, loss_total, "valid")
        
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, dense_info)
