import os
import shutil
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

import taichi as ti
import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
import robohang.policy.policy_utils as policy_utils
from robohang.policy.net import UNetMultiHead, CNN, UNet
import robohang.common.utils as utils
from robohang.policy.funnel.funnel_learn import rotate_and_translate

import lightning.pytorch as pl


@dataclass
class KeypointsDataDict:
    depth_path: str
    mask_path: str
    left_path: str
    right_path: str
    coverage: float


class KeypointsDataset(Dataset):
    def __init__(
        self,
        data_list: List[KeypointsDataDict],
        data_index_table: np.ndarray, 
        is_keep: np.ndarray,
        all_rank: np.ndarray,
        ds_cfg: omegaconf.DictConfig,
        name="none",
    ) -> None:
        super().__init__()

        if name == "train":
            data_index_table_keep = []
            for x in data_index_table:
                if is_keep[x]:
                    data_index_table_keep.append(x)
            self._data_index_table = np.array(data_index_table_keep)
        elif name == "valid":
            self._data_index_table = data_index_table.copy()
        else:
            raise ValueError(name)
        self._name = name

        self._data_list = Manager().list(data_list)
        self._ds_cfg = copy.deepcopy(ds_cfg)

        self._size = int(len(self._data_index_table))
        self._dtype = getattr(np, self._ds_cfg.dtype)

        print(f"funnel dataset {name}: len {self._size}")

        ## for debug
        if utils.ddp_is_rank_0():
            get_item_rank_list = []
            for i in range(self.__len__()):
                get_item_rank_list.append(all_rank[self._data_index_table[i]])
            plt.figure()
            n, bins, patches = plt.hist(get_item_rank_list, bins=20)
            os.makedirs("debug", exist_ok=True)
            plt.savefig(f"debug/{self._name}.png")
            plt.close()
    
    def __len__(self):
        return self._size
    
    def __getitem__(self, index: int):
        # extract data
        data: KeypointsDataDict = self._data_list[self._data_index_table[index]]
        depth_raw: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_raw: np.ndarray = np.load(data.mask_path).astype(self._dtype)
        left_raw: np.ndarray = np.load(data.left_path).astype(self._dtype)
        right_raw: np.ndarray = np.load(data.right_path).astype(self._dtype)

        # augment data
        depth, mask, info = learn_utils.data_augmentation(depth_raw, mask_raw, self._ds_cfg.aug)
        if self._name == "train":
            depth, mask_lr, _, _ = rotate_and_translate(self._ds_cfg.aug, depth, np.array([mask, left_raw, right_raw], self._dtype).transpose(1, 2, 0))
            mask, left, right = mask_lr[:, :, 0], mask_lr[:, :, 1], mask_lr[:, :, 2]
        elif self._name == "valid":
            left = left_raw.copy()
            right = right_raw.copy()
        else:
            raise ValueError(self._name)
        
        return dict(
            weight=1.0, # self.weight[idx],
            depth=depth,
            mask=mask,
            left=left,
            right=right,
            
            depth_raw=depth_raw,
            mask_raw=mask_raw,
            left_raw=left_raw,
            right_raw=right_raw,
            depth_path=os.path.relpath(data.depth_path, utils.get_path_handler()(".")), # self.depth_path[idx]
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
            # state = np.load(os.path.join(base_dir, "state", format_step(depth_idx) + ".npy"), allow_pickle=True).item()

            for batch_idx in range(int(misc_info["batch_size"])):
                if sim_error_batch["all"][batch_idx]: # skip sim_error batch
                    continue

                # obs
                obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))
                depth_path = os.path.join(obs_base_dir, "reproject_depth", format_step(depth_idx) + ".npy")
                mask_path = os.path.join(obs_base_dir, "reproject_is_garment_nointerp", format_step(depth_idx) + ".npy")
                with open(os.path.join(obs_base_dir, "info", format_step(depth_idx) + ".json"), "r") as f_obj:
                    obs_info = json.load(f_obj)
                assert MetaInfo.check_reproject_info(obs_info["reproject"]), str(obs_info["reproject"])

                # gt pixel
                faceup = bool(score_batch["orientation"]["flip_y"][batch_idx] == 0)

                left_path = os.path.join(base_dir, "keypoints", format_step(depth_idx), "left", format_batch(batch_idx) + ".npy")
                right_path = os.path.join(base_dir, "keypoints", format_step(depth_idx), "right", format_batch(batch_idx) + ".npy")
                if (not faceup) and bool(make_cfg.exchange_left_right):
                    left_path, right_path = right_path, left_path # exchange left and right for better labeling
                
                # data weight
                '''weight=(score_batch["coverage"][batch_idx] / 0.5) ** make_cfg.coverage_weight_exp
                if score_batch["coverage"][batch_idx] < make_cfg.coverage_threshold: # use data with large coverage
                    weight = 1e-3'''
                coverage = score_batch["coverage"][batch_idx]

                # assemble data
                all_data_list.append(KeypointsDataDict(
                    depth_path=depth_path, 
                    mask_path=mask_path,
                    left_path=left_path,
                    right_path=right_path, 
                    coverage=coverage,
                ))

    total_size = len(all_data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    
    def get_rank(x: np.ndarray): return np.searchsorted(np.sort(x), x, side="right")
    all_rank = get_rank(np.array([x.coverage for x in all_data_list]))
    is_keep = ((all_rank / len(all_data_list)) ** float(make_cfg.coverage_weight_exp)) >= np.random.rand(len(all_data_list))
    # all_rank and all_data_list is aligned

    trds = KeypointsDataset(all_data_list, path_idx_permutated[valid_size:], is_keep, all_rank, ds_cfg, name=f"train")
    vlds = KeypointsDataset(all_data_list, path_idx_permutated[:valid_size], is_keep, all_rank, ds_cfg, name=f"valid")
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

        kwargs = copy.deepcopy(model_kwargs["net"])
        kwargs["input_channel"] = 2
        kwargs["input_height"] = MetaInfo.reproject_height()
        kwargs["input_width"] = MetaInfo.reproject_width()
        kwargs["segment_output_channels"] = [1, 1]
        self.net = UNetMultiHead(**kwargs)
            
        self.d_unit = float(model_kwargs["d_unit"])
        self.weight_seg_smooth = float(learn_kwargs["weight_seg_smooth"])
        self.weight_cls_smooth = float(learn_kwargs["weight_cls_smooth"])
        self.weight_cls_global = float(learn_kwargs["weight_cls_global"])
        self.weight_cls_start_step = int(learn_kwargs["weight_cls_start_step"])
        self.use_unet_pred_as_gt = bool(learn_kwargs["use_unet_pred_as_gt"])
        self.detach_global_feature = bool(learn_kwargs["detach_global_feature"])
        self.automatic_optimization = False
        
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
        - i: [B, C, H, W]
        """
        B, H, W = depth.shape
        depth = self.normalize_depth(depth)
        return torch.concat([depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)
    
    def extract_data(self, batch):
        depth: torch.Tensor = batch["depth"] # [B, H, W]
        mask: torch.Tensor = batch["mask"] # [B, H, W]
        left: torch.Tensor = batch["left"] # [B, H, W]
        right: torch.Tensor = batch["right"] # [B, H, W]
        weight_cov: torch.Tensor = batch["weight"] # [B, ]

        B, H, W = depth.shape
        assert mask.shape == depth.shape, mask.shape
        assert left.shape == depth.shape, left.shape
        assert right.shape == depth.shape, right.shape
        assert weight_cov.shape == (B, ), weight_cov.shape

        return depth, mask, left, right, weight_cov
    
    def run_decode_seg(self, global_feature: torch.Tensor, xs: List[torch.Tensor]):
        pred_dense: torch.Tensor = self.net.decode_seg(global_feature, xs) # [B, 2, H, W]
        (B, _, H, W), device = pred_dense.shape, pred_dense.device

        pred_ij = torch.argmax(pred_dense.view(B, 2, -1), dim=2)[..., None]
        pred_ij = torch.concat([pred_ij // W, pred_ij % W], dim=2) # [B, 2, 2]

        pred_l_maxval = pred_dense[torch.arange(B).to(device), 0, pred_ij[:, 0, 0], pred_ij[:, 0, 1]] # [B, ]
        pred_r_maxval = pred_dense[torch.arange(B).to(device), 1, pred_ij[:, 1, 0], pred_ij[:, 1, 1]] # [B, ]
        pred_maxval = torch.concat([pred_l_maxval[:, None], pred_r_maxval[:, None]], dim=1) # [B, 2]

        return pred_dense, pred_ij, pred_maxval

    def run_decode_cls(self, global_feature: torch.Tensor):
        B, C, H, W = global_feature.shape
        pred_cls: torch.Tensor = self.net.decode_cls(global_feature)

        assert pred_cls.shape == (B, 1), pred_cls.shape
        pred_cls = pred_cls.view((B, ))

        return pred_cls

    def get_weight_cls_global(self):
        if self.global_step >= self.weight_cls_start_step:
            return self.weight_cls_global
        else:
            return self.weight_cls_global * (self.global_step / self.weight_cls_start_step)

    def forward_all(self, batch):
        depth, mask, left, right, weight_cov = self.extract_data(batch)
        i = self.preprocess_input(depth, mask)
        global_feature, xs = self.net.encode(i)
        pred_dense, pred_ij, pred_maxval = self.run_decode_seg(global_feature, xs)
        if self.detach_global_feature:
            pred_cls = self.run_decode_cls(global_feature.clone().detach())
        else:
            pred_cls = self.run_decode_cls(global_feature)

        GT_TH = 0.5 # interpolation during rotation, gt_seg may be 0.x, use 0.5 as a threshold

        bce = torch.nn.functional.binary_cross_entropy_with_logits
        gt_seg = torch.concat([left.unsqueeze(1), right.unsqueeze(1)], dim=1) # [B, 2, H, W]
        (B, _, H, W), device = gt_seg.shape, gt_seg.device
        
        weight_seg_neg = (1. + self.weight_seg_smooth) / (self.weight_seg_smooth + (gt_seg.view(B, 2, -1) < GT_TH).float().mean(dim=2))
        weight_seg_pos = (1. + self.weight_seg_smooth) / (self.weight_seg_smooth + (gt_seg.view(B, 2, -1) >= GT_TH).float().mean(dim=2))
        weight_seg = torch.where(gt_seg < GT_TH, weight_seg_neg[:, :, None, None], weight_seg_pos[:, :, None, None]) # [B, 2, H, W]
        loss_seg = bce(pred_dense, gt_seg, weight_seg * weight_cov[:, None, None, None])

        # optimization of classifier network
        seg_l_max = gt_seg[torch.arange(B).to(device), 0, pred_ij[:, 0, 0], pred_ij[:, 0, 1]] # [B, ]
        seg_r_max = gt_seg[torch.arange(B).to(device), 1, pred_ij[:, 1, 0], pred_ij[:, 1, 1]] # [B, ]
        segmax_success = torch.logical_and(seg_l_max >= GT_TH, seg_r_max >= GT_TH) # the selected pixels success
        if self.use_unet_pred_as_gt:
            cls_lable_in_loss = segmax_success.float() # [B, ]
        else:
            cls_lable_in_loss = torch.logical_and(
                gt_seg[:, 0, :, :].view(B, -1).max(dim=1).values >= GT_TH, 
                gt_seg[:, 1, :, :].view(B, -1).max(dim=1).values >= GT_TH
            ).float() # [B, ]
        
        weight_cls_neg = (self.weight_cls_smooth + 1.) / (self.weight_cls_smooth + (cls_lable_in_loss == 0.).float().mean())
        weight_cls_pos = (self.weight_cls_smooth + 1.) / (self.weight_cls_smooth + (cls_lable_in_loss == 1.).float().mean())
        weight_cls = torch.where(cls_lable_in_loss == 0., weight_cls_neg, weight_cls_pos)
        loss_cls = bce(pred_cls, cls_lable_in_loss, weight_cls * weight_cov)

        loss_total = loss_cls * self.get_weight_cls_global() + loss_seg
        loss_dict = dict(seg=loss_seg, cls=loss_cls)

        # evaluation of classifier network
        cls_fn = torch.logical_and(cls_lable_in_loss == 1., pred_cls < 0.).float().mean()
        cls_tn = torch.logical_and(cls_lable_in_loss == 0., pred_cls < 0.).float().mean()
        cls_fp = torch.logical_and(cls_lable_in_loss == 0., pred_cls >= 0.).float().mean()
        cls_tp = torch.logical_and(cls_lable_in_loss == 1., pred_cls >= 0.).float().mean()

        clseg_fn = torch.logical_and(segmax_success == 1., pred_cls < 0.).float().mean()
        clseg_tn = torch.logical_and(segmax_success == 0., pred_cls < 0.).float().mean()
        clseg_fp = torch.logical_and(segmax_success == 0., pred_cls >= 0.).float().mean()
        clseg_tp = torch.logical_and(segmax_success == 1., pred_cls >= 0.).float().mean()

        gt_success = torch.logical_and(
            gt_seg[:, 0, :, :].view(B, -1).max(dim=1).values >= GT_TH, 
            gt_seg[:, 1, :, :].view(B, -1).max(dim=1).values >= GT_TH, 
        ) # there exist pixels which can succeed
        max_err = torch.logical_and(gt_success, torch.logical_not(segmax_success))

        segmax_pred = torch.logical_and(pred_maxval[:, 0] >= 0., pred_maxval[:, 1] >= 0.)
        segmax_tp = torch.logical_and(segmax_pred, segmax_success)
        segmax_fp = torch.logical_and(segmax_pred, torch.logical_not(segmax_success))

        err_dict = dict(
            cls=cls_fn+cls_fp, 
            seg=torch.mean(torch.logical_xor(gt_seg < GT_TH, pred_dense < 0.).float()),
        )

        met_dict = dict(
            cls_fn=cls_fn,
            cls_tn=cls_tn,
            cls_fp=cls_fp,
            cls_tp=cls_tp,
            clseg_fn=clseg_fn,
            clseg_tn=clseg_tn,
            clseg_fp=clseg_fp,
            clseg_tp=clseg_tp,
            segmax_err=max_err.float().mean(),
            segmax_success=segmax_success.float().mean(),
            segmax_tp=segmax_tp.float().mean(),
            segmax_fp=segmax_fp.float().mean(),
        )

        dense_info = dict(
            dense=pred_dense,
            gt_seg=gt_seg,
            depth=depth,
            mask=mask,
            ij_left=pred_ij[:, 0, :],
            ij_right=pred_ij[:, 1, :],
            pred_cls=pred_cls,
            gt_cls=cls_lable_in_loss,
            weight_cov=weight_cov,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],
            left_raw=batch["left_raw"],
            right_raw=batch["right_raw"],
            depth_path=batch["depth_path"],
        )
        return loss_dict, err_dict, met_dict, loss_total, dense_info
    
    def log_all(self, loss_dict: Dict[str, torch.Tensor], err_dict: Dict[str, torch.Tensor], met_dict: Dict[str, torch.Tensor], loss_total: torch.Tensor, name: str):
        for k in err_dict.keys():
            self.log(f"{name}/err_{k}", err_dict[k].detach().clone(), sync_dist=True)
        for k in loss_dict.keys():
            self.log(f"{name}/loss_{k}", loss_dict[k].detach().clone(), sync_dist=True)
        for k in met_dict.keys():
            self.log(f"{name}/met_{k}", met_dict[k].detach().clone(), sync_dist=True)
        self.log_dict({f"{name}/total_loss": loss_total.detach().clone()}, sync_dist=True)
        self.log("weight_cls_global", self.get_weight_cls_global())
    
    def _depth_overlay(self, depth: np.ndarray, left: np.ndarray, right: np.ndarray):
        depth = (depth - depth.min(axis=(1, 2), keepdims=True)) / \
            (depth.max(axis=(1, 2), keepdims=True) - depth.min(axis=(1, 2), keepdims=True) + 1e-5) # [B, H, W]
        depth = np.where(np.logical_or(left > 0., right > 0.), 0., depth)
        depth = np.repeat(depth[..., None], 3, axis=3)
        depth[..., 0] += left
        depth[..., 1] += right
        return depth

    def log_img(self, batch_idx, dense_info):
        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]
        act_str_raw = []
        pred_str = []
        for i in range(plot_batch_size):
            path_full = dense_info["depth_path"][i]
            path_with_endline = []
            line_width = 20
            for s in range((len(path_full) + line_width - 1) // line_width):
                path_with_endline.append(path_full[s * line_width: min((s + 1) * line_width, len(path_full))])
            act_str_raw.append("\n".join(path_with_endline))
            pred_str.append(
                f"weight={dense_info['weight_cov'][i]:.4f}\n"
                f"pred={dense_info['pred_cls'][i]:.4f} / cls={dense_info['gt_cls'][i]:.4f}\n"
                f"red=left green=right"
            )
        depth_overlay_kp = self._depth_overlay(
            utils.torch_to_numpy(dense_info["depth_raw"]),
            utils.torch_to_numpy(dense_info["left_raw"]), 
            utils.torch_to_numpy(dense_info["right_raw"]),
        )

        learn_utils.plot_wrap(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(dense_info["mask_raw"]),
                depth_overlay_kp,
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"]),
                utils.torch_to_numpy(dense_info["dense"][:, 0, :, :]),
                utils.torch_to_numpy(dense_info["dense"][:, 1, :, :]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(dense_info["depth_raw"])[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(dense_info["ij_left"].to(torch.long)),
                        utils.torch_to_numpy(dense_info["ij_right"].to(torch.long)),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                    widths=[3, 3],
                ),
            ],
            tag=f"dense_predict/{batch_idx}",
            titles=[act_str_raw, "mask_raw", "red=left green=right", "depth_nmd", "mask", "pred_left", "pred_right", pred_str],
            colorbars=["gist_gray", "gist_gray", None, "gist_gray", "gist_gray", "viridis", "viridis", None],
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )
    
    def log_img_eval(self, depth: torch.Tensor, mask: torch.Tensor, pred_dense: torch.Tensor, pred_ij: torch.Tensor):
        B = depth.shape[0]
        pred_str = []
        for i in range(B):
            pred_str.append(f"keypoints red=left, green=right")
        
        return learn_utils.plot_wrap_fig(
            denses=[
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(depth)[..., None], 3, axis=3),
                    actions=[
                        utils.torch_to_numpy(pred_ij[:, 0, :]).astype(np.int32),
                        utils.torch_to_numpy(pred_ij[:, 1, :]).astype(np.int32),
                    ],
                    colors=[np.array([1., 0., 0.]), np.array([0., 1., 0.])],
                    widths=[3, 3],
                ), 
                utils.torch_to_numpy(mask), 
                utils.torch_to_numpy(pred_dense[:, 0, :, :]), 
                utils.torch_to_numpy(pred_dense[:, 1, :, :])
            ],
            titles=[pred_str, "mask", "left", "right"],
            colorbars=[None, "gist_gray", "viridis", "viridis"],
            plot_batch_size=1,
        )

    def training_step(self, batch, batch_idx):
        opt:torch.optim.Optimizer = self.optimizers()
        sch:torch.optim.lr_scheduler.LRScheduler = self.lr_schedulers()

        loss_dict, err_dict, met_dict, loss_total, dense_info = self.forward_all(batch)
        self.log_all(loss_dict, err_dict, met_dict, loss_total, "train")
        
        opt.zero_grad()
        self.manual_backward(loss_total)
        opt.step()
        sch.step()

    def validation_step(self, batch, batch_idx):
        loss_dict, err_dict, met_dict, loss_total, dense_info = self.forward_all(batch)

        if batch_idx > 0:
            self.log_all(loss_dict, err_dict, met_dict, loss_total, "valid")
        
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, dense_info)


class KeypointsModuleSinglehead(KeypointsModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        raise NotImplementedError
        super().__init__()
        self.save_hyperparameters()
        
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.actor = UNetMultiHead(
            input_channel=2, input_height=MetaInfo.reproject_height(), input_width=MetaInfo.reproject_width(),
            segment_output_channels=[1, 1], classif_output_mlp_dims=[], **(model_kwargs["net"]["actor"])
        )
        self.critic_l = CNN(
            input_channel=3, input_height=MetaInfo.reproject_height(), input_width=MetaInfo.reproject_width(),
            output_channel=1, **(model_kwargs["net"]["critic"]),
        )
        self.critic_r = CNN(
            input_channel=3, input_height=MetaInfo.reproject_height(), input_width=MetaInfo.reproject_width(),
            output_channel=1, **(model_kwargs["net"]["critic"]),
        )
            
        self.d_unit = float(model_kwargs["d_unit"])
        self.weight_seg_smooth = float(learn_kwargs["weight_seg_smooth"])
        self.weight_cls_smooth = float(learn_kwargs["weight_cls_smooth"])
        self.automatic_optimization = False
        
    def configure_optimizers(self):
        optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
                torch.nn.ModuleList([self.critic_l, self.critic_r, self.actor]).parameters(),
                **(self.learn_kwargs["optimizer"]["cfg"]),
            ),
        ]
        schedule = [
            getattr(torch.optim.lr_scheduler, self.learn_kwargs["schedule"]["name"])(
                optimizer[0],
                **(self.learn_kwargs["schedule"]["cfg"]),
            ),
        ]
        '''optimizer = [
            getattr(torch.optim, self.learn_kwargs["optimizer"]["name"])(
                self.actor.parameters(),
                **(self.learn_kwargs["optimizer"]["cfg"]),
            ), torch.optim.SGD(torch.nn.ModuleList([self.critic_l, self.critic_r]).parameters(), lr=1e-4)
        ]
        schedule = [
            getattr(torch.optim.lr_scheduler, self.learn_kwargs["schedule"]["name"])(
                optimizer[0],
                **(self.learn_kwargs["schedule"]["cfg"]),
            ), torch.optim.lr_scheduler.MultiStepLR(optimizer[1], [])
        ]'''
        return optimizer, schedule
    
    def run_predict(self, i: torch.Tensor):
        (B, C, H, W), dtype, device = i.shape, i.dtype, i.device

        pred_dense: torch.Tensor = self.actor(i)["seg"] # [B, 2, H, W]
        assert pred_dense.shape == (B, 2, H, W), pred_dense.shape
        pred_ij = torch.argmax(pred_dense.view(B, 2, -1), dim=2)[..., None]
        pred_ij = torch.concat([pred_ij // W, pred_ij % W], dim=2) # [B, 2, 2]

        pred_ij_l_onehot = learn_utils.one_hot(pred_ij[:, 0, :], (B, H, W), dtype, device)
        pred_ij_r_onehot = learn_utils.one_hot(pred_ij[:, 1, :], (B, H, W), dtype, device)

        pred_cls_l: torch.Tensor = self.critic_l(torch.concat([i, pred_ij_l_onehot.unsqueeze(1)], dim=1)).squeeze(1)
        assert pred_cls_l.shape == (B, ), pred_cls_l.shape
        pred_cls_r: torch.Tensor = self.critic_r(torch.concat([i, pred_ij_r_onehot.unsqueeze(1)], dim=1)).squeeze(1)
        assert pred_cls_r.shape == (B, ), pred_cls_r.shape

        return pred_dense, pred_ij, pred_cls_l, pred_cls_r
    
    def forward_all(self, batch):
        depth, mask, left, right, weight_cov = self.extract_data(batch)
        i = self.preprocess_input(depth, mask)
        pred_dense, pred_ij, pred_cls_l, pred_cls_r = self.run_predict(i)

        GT_TH = 0.5 # interpolation during rotation, gt_seg may be 0.x, use 0.5 as a threshold

        bce = torch.nn.functional.binary_cross_entropy_with_logits
        gt_seg = torch.concat([left.unsqueeze(1), right.unsqueeze(1)], dim=1) # [B, 2, H, W]
        (B, _, H, W), device = gt_seg.shape, gt_seg.device
        
        weight_seg_neg = (1. + self.weight_seg_smooth) / (self.weight_seg_smooth + (gt_seg.view(B, 2, -1) < GT_TH).float().mean(dim=2))
        weight_seg_pos = (1. + self.weight_seg_smooth) / (self.weight_seg_smooth + (gt_seg.view(B, 2, -1) >= GT_TH).float().mean(dim=2))
        weight_seg = torch.where(gt_seg < GT_TH, weight_seg_neg[:, :, None, None], weight_seg_pos[:, :, None, None]) # [B, 2, H, W]
        loss_seg = bce(pred_dense, gt_seg, weight_seg * weight_cov[:, None, None, None])

        gt_cls_l = gt_seg[torch.arange(B).to(device), 0, pred_ij[:, 0, 0], pred_ij[:, 0, 1]] # [B, ]
        gt_cls_r = gt_seg[torch.arange(B).to(device), 1, pred_ij[:, 1, 0], pred_ij[:, 1, 1]] # [B, ]
        
        loss_total = loss_seg
        loss_dict = dict(seg=loss_seg)
        err_dict = dict(seg=torch.mean(torch.logical_xor(gt_seg < GT_TH, pred_dense < 0.).float()))
        for lr_idx, lr_str in enumerate(["cls_l", "cls_r"]):
            weight_cls_neg = (self.weight_cls_smooth + 1.) / (self.weight_cls_smooth + ([gt_cls_l, gt_cls_r][lr_idx] < GT_TH).float().mean())
            weight_cls_pos = (self.weight_cls_smooth + 1.) / (self.weight_cls_smooth + ([gt_cls_l, gt_cls_r][lr_idx] >= GT_TH).float().mean())
            weight_cls = torch.where([gt_cls_l, gt_cls_r][lr_idx] < GT_TH, weight_cls_neg, weight_cls_pos)
            loss_cls = bce([pred_cls_l, pred_cls_r][lr_idx], [gt_cls_l, gt_cls_r][lr_idx], weight_cls * weight_cov)
            loss_total = loss_total + loss_cls
            loss_dict[lr_str] = loss_cls
            err_dict[lr_str] = torch.logical_xor([pred_cls_l, pred_cls_r][lr_idx] >= 0., [gt_cls_l, gt_cls_r][lr_idx] >= GT_TH).float().mean()
        
        pred_cls_bin = torch.logical_and(pred_cls_l >= 0., pred_cls_r >= 0.)
        gt_cls_bin = torch.logical_and(gt_cls_l >= GT_TH, gt_cls_r >= GT_TH)
        cls_tp = torch.logical_and(gt_cls_bin, pred_cls_bin).float().mean()
        cls_fn = torch.logical_and(gt_cls_bin, torch.logical_not(pred_cls_bin)).float().mean()
        cls_fp = torch.logical_and(torch.logical_not(gt_cls_bin), pred_cls_bin).float().mean()
        cls_tn = torch.logical_and(torch.logical_not(gt_cls_bin), torch.logical_not(pred_cls_bin)).float().mean()
        
        err_dict["cls"] = cls_fn + cls_fp
        met_dict = dict(
            cls_fn=cls_fn,
            cls_tn=cls_tn,
            cls_fp=cls_fp,
            cls_tp=cls_tp,
        )

        dense_info = dict(
            dense=pred_dense,
            gt_seg=gt_seg,
            depth=depth,
            mask=mask,
            ij_left=pred_ij[:, 0, :],
            ij_right=pred_ij[:, 1, :],
            pred_cls_l=pred_cls_l,
            pred_cls_r=pred_cls_r,
            gt_cls=gt_cls_bin,
            weight_cov=weight_cov,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],
            left_raw=batch["left_raw"],
            right_raw=batch["right_raw"],
            depth_path=batch["depth_path"],
        )
        return loss_dict, err_dict, met_dict, loss_total, dense_info

    def training_step(self, batch, batch_idx):
        opt1 = self.optimizers()
        sch1 = self.lr_schedulers()

        loss_dict, err_dict, met_dict, loss_total, dense_info = self.forward_all(batch)
        self.log_all(loss_dict, err_dict, met_dict, loss_total, "train")
        
        opt1.zero_grad()
        self.manual_backward(loss_total)
        opt1.step()
        sch1.step()