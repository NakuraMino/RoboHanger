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
from robohang.policy.funnel.funnel_learn import generate_positional_encoding, MetaInfo, rotate_and_translate

import lightning.pytorch as pl


@dataclass
class CoverageDataDict:
    depth_path: str
    mask_path: str
    coverage: float


class CoverageDataset(Dataset):
    def __init__(
        self,
        data_list:List[CoverageDataDict],
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
        data: CoverageDataDict = self._data_list[self._data_index_table[index]]
        depth_raw: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_raw: np.ndarray = np.load(data.mask_path).astype(self._dtype)
        
        coverage: float = data.coverage

        # positional encoding
        pe = generate_positional_encoding(depth_raw.shape[0], depth_raw.shape[1], self._dtype)

        # augment data
        depth, mask, info = learn_utils.data_augmentation(depth_raw, mask_raw, self._ds_cfg.aug)
        if self._name == "train":
            depth, mask, pe, _, _ = rotate_and_translate(self._ds_cfg.aug, depth, mask, pe)
        elif self._name == "valid":
            pass
        else:
            raise ValueError(self._name)
        
        return dict(
            depth=depth,
            mask=mask,
            pe=pe,
            coverage=coverage,
            
            depth_raw=depth_raw,
            mask_raw=mask_raw,
            depth_path=data.depth_path,
        )


def make_coverage_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    make_cfg: omegaconf.DictConfig,
    ds_cfg: omegaconf.DictConfig,
):
    pattern = "^completed.txt$"
    df = learn_utils.DataForest(data_path, [pattern])
    node_n_raw = df.get_forest_size(pattern)

    all_data_list: List[CoverageDataDict] = []
    
    print("scanning all trajectories ...")
    for idx in tqdm.tqdm(range(node_n_raw)):
        misc_path = os.path.join(os.path.dirname(df.get_item(pattern, idx).file_path), "misc.json")
        with open(misc_path, "r") as f_obj:
            misc_info: dict = json.load(f_obj)
        base_dir = os.path.dirname(misc_path)

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
            for batch_idx in range(int(misc_info["batch_size"])):
                if sim_error_batch["all"][batch_idx]: # skip sim_error batch
                    continue

                # obs
                obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))
                depth_path = os.path.join(obs_base_dir, "reproject_depth", format_step(depth_idx) + ".npy")
                mask_path = os.path.join(obs_base_dir, "reproject_is_garment", format_step(depth_idx) + ".npy")
                with open(os.path.join(obs_base_dir, "info", format_step(depth_idx) + ".json"), "r") as f_obj:
                    obs_info = json.load(f_obj)
                assert MetaInfo.check_reproject_info(obs_info["reproject"]), str(obs_info["reproject"])

                # gt pixel
                coverage = float(score_batch["coverage"][batch_idx])

                # assemble data
                all_data_list.append(CoverageDataDict(
                    depth_path=depth_path, 
                    mask_path=mask_path,
                    coverage=coverage,
                ))

    total_size = len(all_data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    trds = CoverageDataset(all_data_list, path_idx_permutated[valid_size:], ds_cfg, name=f"train")
    vlds = CoverageDataset(all_data_list, path_idx_permutated[:valid_size], ds_cfg, name=f"valid")
    return trds, vlds


class CoverageModule(pl.LightningModule):
    pred_target_str = ["coverage"]
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.nets = torch.nn.ModuleDict({})
        for k in self.pred_target_str:
            kwargs = copy.deepcopy(model_kwargs["net"])
            kwargs["input_channel"] = 2 + int(model_kwargs["use_pe"]) * 2
            kwargs["input_height"] = MetaInfo.reproject_height()
            kwargs["input_width"] = MetaInfo.reproject_width()
            if k == "coverage":
                kwargs["output_channel"] = 1
            else:
                raise ValueError(k)
            self.nets[k] = CNN(**kwargs)
            
        self.d_unit = float(model_kwargs["d_unit"])

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
    
    def preprocess_input(self, depth: torch.Tensor, mask: torch.Tensor, pe: torch.Tensor):
        """
        `d = (d - mean) / unit`

        Args:
        - depth: [B, H, W]
        - mask: [B, H, W]
        - pe: [B, 2, H, W]

        Return:
        - i: [B, C, H, W]
        """
        B, H, W = depth.shape
        depth = self.normalize_depth(depth)
        if self.model_kwargs["use_pe"]:
            return torch.concat([depth.unsqueeze(1), mask.unsqueeze(1), pe], dim=1)
        else:
            return torch.concat([depth.unsqueeze(1), mask.unsqueeze(1)], dim=1)
    
    def extract_data(self, batch):
        """
        Return:
        - depth: [B, H, W]
        - mask: [B, H, W]
        - action: [B, 4]
        - reward: dict, str -> [B]
        - pe: [B, 2, H, W]
        """
        depth: torch.Tensor = batch["depth"] # [B, H, W]
        mask: torch.Tensor = batch["mask"] # [B, H, W]
        coverage: torch.Tensor = batch["coverage"] # [B], float
        pe: torch.Tensor = batch["pe"] # [B, 2, H, W]

        B, H, W = depth.shape
        assert mask.shape == depth.shape, mask.shape
        assert coverage.shape == (B, ), coverage.shape
        assert pe.shape == (B, 2, H, W), pe.shape

        return depth, mask, coverage, pe
    
    def forward_all(self, batch):
        depth, mask, coverage, pe = self.extract_data(batch)
        i = self.preprocess_input(depth, mask, pe)

        pred: Dict[str, torch.Tensor] = {}
        for k in self.pred_target_str:
            o: torch.Tensor = self.nets[k](i)
            if k == "coverage":
                assert len(o.shape) == 2 and o.shape[1] == 1, o.shape
                o = o.view(o.shape[0])
            else:
                raise ValueError(k)
            pred[k] = o
        
        loss_dict = {}
        err_dict = {}
        loss_total = None
        for k in self.pred_target_str:
            if k == "coverage":
                loss_single: torch.Tensor = torch.nn.functional.l1_loss(pred[k], coverage) # scalar
                err_single: torch.Tensor = torch.abs(pred[k] - coverage).mean() # scalar
            else:
                raise ValueError(k)
            
            loss_dict[k] = loss_single
            err_dict[k] = err_single
            if loss_total is None:
                loss_total = loss_single
            else:
                loss_total = loss_total + loss_single
        
        dense_info = dict(
            pred=pred,
            depth=depth,
            mask=mask,
            coverage=coverage,

            depth_raw=batch["depth_raw"],
            mask_raw=batch["mask_raw"],
            depth_path=batch["depth_path"],
        )
        return loss_dict, err_dict, loss_total, dense_info
    
    def log_all(self, loss_dict: Dict[str, torch.Tensor], err_dict: Dict[str, torch.Tensor], loss_total: torch.Tensor, name: str):
        for k in self.pred_target_str:
            self.log_dict({
                f"{name}/{k}_loss": loss_dict[k].detach().cpu(),
                f"{name}/{k}_err": err_dict[k].detach().cpu(),
            })
        self.log_dict({f"{name}/total_loss": loss_total.detach().cpu()})
    
    def log_img(self, batch_idx, dense_info):
        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]
        act_str_raw = []
        act_str_aug = []
        for i in range(plot_batch_size):
            path_full = dense_info["depth_path"][i]
            path_with_endline = []
            line_width = 30
            for s in range((len(path_full) + line_width - 1) // line_width):
                path_with_endline.append(path_full[s * line_width: min((s + 1) * line_width, len(path_full))])
            act_str_raw.append(
                f"gt raw red=left green=right\n"
                f"coverage={dense_info['coverage'][i]:.4f} "
                f"pred={dense_info['pred']['coverage'][i]:.4f}"
            )
            act_str_aug.append(
                "gt red=left green=right\n" + 
                "\n".join(path_with_endline)
            )

        learn_utils.plot_wrap(
            denses=[
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(dense_info["mask_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
                utils.torch_to_numpy(dense_info["mask"]),
                utils.torch_to_numpy(dense_info["depth_raw"]),
                utils.torch_to_numpy(self.normalize_depth(dense_info["depth"])),
            ],
            tag=f"dense_predict/{batch_idx}",
            titles=["depth_raw", "mask_raw", "depth_nmd", "mask", act_str_raw, act_str_aug],
            colorbars=["gist_gray", "gist_gray", "gist_gray", "gist_gray", None, None],
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
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
