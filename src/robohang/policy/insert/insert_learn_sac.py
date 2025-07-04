import os
import copy
from typing import List, Dict, Literal, Union, Tuple, Any, Optional
import json
import math
from dataclasses import dataclass
import pprint

import tqdm
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

import lightning.pytorch as pl

import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
import robohang.policy.policy_utils as policy_utils
from robohang.policy.net import DoubleConv, Down, CNN
import robohang.common.utils as utils

from robohang.policy.insert.insert_learn import concat_mask, REWARD_THRESHOLD
from robohang.policy.insert.insert_policy import InsertPolicy, prepare_obs


ACTION_Z = 0.68
ACTION_D = 1.82

def action_xy_to_ij(action: torch.Tensor, height: int, width: int):
    device = action.device
    action_np = utils.torch_to_numpy(action)
    action_ij = []
    for x, y in action_np:
        i, j = policy_utils.xyz2ij(x, y, ACTION_Z, MetaInfo.reproject_camera_info)
        action_ij.append([max(min(i, height - 1), 0), max(min(j, width - 1), 0)])
    return torch.tensor(action_ij, dtype=torch.long, device=device)


def action_ij_to_xy(action: torch.Tensor):
    device = action.device
    action_np = utils.torch_to_numpy(action)
    action_xy = []
    for i, j in action_np:
        x, y = policy_utils.ijd2xyz(i, j, ACTION_D, MetaInfo.reproject_camera_info)
        action_xy.append([x, y])
    return torch.tensor(action_xy, dtype=torch.float32, device=device)


#https://github.com/pranz24/pytorch-soft-actor-critic

LOG_SIG_MAX = 2
LOG_SIG_MIN = -20
epsilon = 1e-6

class SACQNetwork(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.qf_dist_smooth = float(kwargs.pop("qf_dist_smooth"))
        self.step_encode_bias = float(kwargs.pop("step_encode_bias"))
        self.net1 = CNN(**kwargs)
        self.net2 = CNN(**kwargs)

        ij_to_xy = []
        H, W = self.net1.input_height, self.net1.input_width
        for i in range(H):
            for j in range(W):
                ij_to_xy.append(policy_utils.ijd2xyz(i, j, ACTION_D, MetaInfo.reproject_camera_info)[:2])
        self.ij_to_xy = torch.tensor(np.array(ij_to_xy), dtype=torch.float32, device="cpu").view(H, W, 2)
        
    def forward(self, state: torch.Tensor, action: torch.Tensor, step_idx: torch.Tensor):
        """
        Args:
        - state: [B, C, H, W]
        - action: [B, 2]
        - step_idx: [B, ]
        """
        (B, C, H, W), device = state.shape, state.device
        self.ij_to_xy = self.ij_to_xy.to(device)

        dist = action[:, None, None, :] - self.ij_to_xy[None, :, :, :] # [B, H, W, 2]
        dist_inv = (self.qf_dist_smooth / (dist.norm(dim=3) + self.qf_dist_smooth))[:, None, :, :]
        i = torch.concat([state, (step_idx + self.step_encode_bias)[:, None, None, None].repeat(1, 1, H, W), dist_inv], dim=1)
        x1 = self.net1(i)
        x2 = self.net2(i)

        assert x1.shape == (B, 1), x1.shape
        assert x2.shape == (B, 1), x2.shape
        return x1[:, 0], x2[:, 0]


class SACPolicy(torch.nn.Module):
    @staticmethod
    def compute_scale(step_idx: int) -> List[float]:
        action_space = [MetaInfo.press_action_space, MetaInfo.lift_action_space, MetaInfo.drag_action_space, MetaInfo.rotate_action_space][step_idx]
        return (np.array(action_space.max) - np.array(action_space.min)) / 2
    
    @staticmethod
    def compute_bias(step_idx: int) -> List[float]:
        action_space = [MetaInfo.press_action_space, MetaInfo.lift_action_space, MetaInfo.drag_action_space, MetaInfo.rotate_action_space][step_idx]
        return (np.array(action_space.max) + np.array(action_space.min)) / 2
    
    def __init__(self, **kwargs):
        super().__init__()
        self.step_encode_bias = float(kwargs.pop("step_encode_bias"))
        self.net = CNN(**kwargs) # [mu_x, mu_y, std_x, std_y]

        self.action_scale = torch.tensor(np.array([
            self.compute_scale(step_idx) for step_idx in range(4)
        ]), dtype=torch.float32) # [4, 2]
        self.action_bias = torch.tensor(np.array([
            self.compute_bias(step_idx) for step_idx in range(4)
        ]), dtype=torch.float32) # [4, 2]

    def predict_mean_log_std(self, state: torch.Tensor, step_idx: torch.Tensor):
        B, C, H, W = state.shape
        i = torch.concat([state, (step_idx + self.step_encode_bias)[:, None, None, None].repeat(1, 1, H, W)], dim=1)
        out = self.net(i)
        assert out.shape == (B, 4)

        mean = out[:, 0:2]
        log_std = out[:, 2:4]
        log_std = torch.clamp(log_std, min=LOG_SIG_MIN, max=LOG_SIG_MAX)
        return mean, log_std

    def sample(self, state: torch.Tensor, step_idx: torch.Tensor):
        device = state.device
        B, C, H, W = state.shape

        self.to(device)
        self.action_bias = self.action_bias.to(device)
        self.action_scale = self.action_scale.to(device)
        action_bias = self.action_bias[step_idx, :].clone()
        action_scale = self.action_scale[step_idx, :].clone()
        assert action_bias.shape == action_scale.shape == (B, 2), (action_bias.shape, action_scale.shape)

        mean, log_std = self.predict_mean_log_std(state, step_idx)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * action_scale + action_bias
        log_prob: torch.Tensor = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(action_scale * (1 - y_t.pow(2)) + epsilon)
        log_prob = log_prob.sum(1, keepdim=False)
        mean = torch.tanh(mean) * action_scale + action_bias

        assert action.shape == (B, 2), action.shape
        assert log_prob.shape == (B, ), log_prob.shape
        assert mean.shape == (B, 2), mean.shape
        return action, log_prob, mean


def soft_update(target: torch.nn.Module, source: torch.nn.Module, tau: float):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target: torch.nn.Module, source: torch.nn.Module):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class SACModule(pl.LightningModule):
    def __init__(
        self, 
        model_kwargs: Dict[str, Any], 
        learn_kwargs: Dict[str, Any],
    ):
        super().__init__()
        self.save_hyperparameters()

        self.model_kwargs = copy.deepcopy(model_kwargs)
        self.learn_kwargs = copy.deepcopy(learn_kwargs)

        self.d_unit = self.model_kwargs["d_unit"]
        self.gamma = learn_kwargs["gamma"]
        self.tau = learn_kwargs["tau"]
        self.alpha = learn_kwargs["alpha"]
        self.target_update_interval = learn_kwargs["target_update_interval"]
        self.lr = learn_kwargs["lr"]
        self.weight_decay = learn_kwargs["weight_decay"]

        QNetwork_kwargs = dict(
            input_channel=6,
            input_height=128,
            input_width=128,
            output_channel=1,
            channels_list=model_kwargs["critic"]["channels_list"],
            output_mlp_hidden=model_kwargs["critic"]["output_mlp_hidden"],

            qf_dist_smooth=model_kwargs["qf_dist_smooth"],
            step_encode_bias=model_kwargs["step_encode_bias"],
        )
        SACPolicy_kwargs = dict(
            input_channel=5,
            input_height=128,
            input_width=128,
            output_channel=4,
            channels_list=model_kwargs["policy"]["channels_list"],
            output_mlp_hidden=model_kwargs["policy"]["output_mlp_hidden"],

            step_encode_bias=model_kwargs["step_encode_bias"],
        )

        self.critic = SACQNetwork(**QNetwork_kwargs)
        self.critic_target = SACQNetwork(**QNetwork_kwargs)
        self.policy = SACPolicy(**SACPolicy_kwargs)
        hard_update(self.critic_target, self.critic)

        self.automatic_optimization = False
    
    def configure_optimizers(self):
        optimizer = [
            torch.optim.AdamW(self.critic.parameters(), lr=self.lr, weight_decay=self.weight_decay),
            torch.optim.AdamW(self.policy.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        ]
        return optimizer
    
    def _normalize_depth(self, depth: torch.Tensor):
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
        depth = self._normalize_depth(depth)
        return torch.concat([depth.unsqueeze(1), mask], dim=1)

    def forward_critic(self, batch: Dict[str, torch.Tensor]):
        depth, mask, action, reward, depth_next, mask_next, done, step_idx = (
            batch["depth"], batch["mask"], batch["action"], batch["reward"], 
            batch["depth_next"], batch["mask_next"], batch["done"], batch["step_idx"]
        )
        state = self._preprocess_input(depth, mask)
        state_next = self._preprocess_input(depth_next, mask_next)

        self.policy.eval()
        action_next, next_state_log_pi, _ = self.policy.sample(state_next, step_idx)
        if self.training:
            self.policy.train()
        
        self.critic_target.eval()
        with torch.no_grad():
            qf1_next_target, qf2_next_target = self.critic_target(state_next, action_next, step_idx)
            min_qf_next_target = torch.min(qf1_next_target, qf2_next_target) - self.alpha * next_state_log_pi
            next_q_value = reward + (1. - done) * self.gamma * min_qf_next_target
            assert len(next_q_value.shape) == 1, (next_q_value.shape, reward.shape, done.shape, min_qf_next_target.shape, next_state_log_pi.shape)
        
        qf1, qf2 = self.critic(state, action, step_idx)
        assert len(qf1.shape) == 1, qf1.shape
        assert len(qf2.shape) == 1, qf2.shape
        qf_loss = torch.nn.functional.mse_loss(qf1, next_q_value) + torch.nn.functional.mse_loss(qf2, next_q_value)

        log_img_info = dict()
        misc_info = dict(
            next_q_value=next_q_value.mean(),
        )
        return qf_loss, log_img_info, misc_info

    def forward_policy(self, batch: Dict[str, torch.Tensor]):
        depth, mask, action, reward, depth_next, mask_next, done, step_idx = (
            batch["depth"], batch["mask"], batch["action"], batch["reward"], 
            batch["depth_next"], batch["mask_next"], batch["done"], batch["step_idx"]
        )
        state = self._preprocess_input(depth, mask)
        state_next = self._preprocess_input(depth_next, mask_next)

        pi, log_prob_pi, pi_mean = self.policy.sample(state, step_idx)
        qf1_pi, qf2_pi = self.critic(state, pi, step_idx)
        min_qf_pi = torch.min(qf1_pi, qf2_pi)
        policy_loss = torch.mean(self.alpha * log_prob_pi - min_qf_pi)

        log_img_info = dict(
            state=state, state_next=state_next, 
            pi_mean=pi_mean, pi=pi, action=action, 
            reward=reward, step_idx=step_idx,
        )
        misc_info = dict(
            alpha=torch.tensor(self.alpha), log_prob_pi=log_prob_pi.mean(), 
            min_qf_pi=min_qf_pi.mean()
        )
        return policy_loss, log_img_info, misc_info
    
    def _visualize_qf(self, qf: SACQNetwork, state: torch.Tensor, step_idx: torch.Tensor):
        """return [B, H, W]"""
        (B, C, H, W), device, dtype = state.shape, state.device, state.dtype
        qf.eval()

        div = 4
        result = torch.zeros((B, H // div, W // div), dtype=dtype, device=device)
        for i in range(H // div):
            for j in range(W // div):
                x, y, _ = policy_utils.ijd2xyz((i + 0.5) * div, (j + 0.5) * div, ACTION_D, MetaInfo.reproject_camera_info)
                action = torch.tensor([[x, y]] * B, dtype=dtype, device=device)
                with torch.no_grad():
                    result[:, i, j] = torch.min(*(qf.forward(state, action, step_idx)))
        
        if self.training:
            qf.train()
        
        return result
    
    def log_img(self, batch_idx: int, log_img_info: Dict[str, torch.Tensor]):
        B, C, H, W = log_img_info["state"].shape
        plot_batch_size = self.learn_kwargs["valid"]["plot_dense_predict_num"]

        action_ij = action_xy_to_ij(log_img_info["action"], H, W)
        action_ij_pi = action_xy_to_ij(log_img_info["pi"], H, W)
        action_ij_pi_mean = action_xy_to_ij(log_img_info["pi_mean"], H, W)
        action_str = [
            f"data (red), pi (green), pi mean (blue)\n" + 
            f"step idx {log_img_info['step_idx'][i]}, reward {log_img_info['reward'][i]}\n" + 
            f"data {log_img_info['action'][i, 0]:.4f} {log_img_info['action'][i, 1]:.4f}\n" + 
            f"pi {log_img_info['pi'][i, 0]:.4f} {log_img_info['pi'][i, 1]:.4f}\n" + 
            f"pi_mean {log_img_info['pi_mean'][i, 0]:.4f} {log_img_info['pi_mean'][i, 1]:.4f}\n"
            for i in range(plot_batch_size)
        ]
        qf = self._visualize_qf(self.critic, log_img_info["state"][:plot_batch_size, ...], log_img_info["step_idx"][:plot_batch_size, ...])

        learn_utils.plot_wrap(
            denses=[
                *[utils.torch_to_numpy(log_img_info["state"][:, c, :, :]) for c in range(4)],
                *[utils.torch_to_numpy(log_img_info["state_next"][:, c, :, :]) for c in range(4)],
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(log_img_info["state"][:, 0, :, :])[..., None], 3, axis=3),
                    actions=[action_ij, action_ij_pi, action_ij_pi_mean],
                    colors=[(1., 0., 0.), (0., 1., 0.), (0., 0., 1.)],
                    widths=[3, 3, 3],
                ),
                utils.torch_to_numpy(qf),
            ],
            tag=f"sac",
            titles=["depth", "mask_garment", "mask_inverse", "mask_hanger"] * 2 + [action_str, "qf"],
            colorbars=["gist_gray"] * 8 + [None, "viridis"],
            plot_batch_size=plot_batch_size,
            global_step=self.global_step,
            writer=self.logger.experiment,
        )
    
    def run_predict(self, depth: torch.Tensor, mask: torch.Tensor, step_idx: torch.Tensor):
        """return [B, 2]"""
        state = self._preprocess_input(depth, mask)
        _, _, action = self.policy.sample(state, step_idx)
        log_img_info = dict(state=state, action=action, step_idx=step_idx)
        return action, log_img_info
    
    def log_img_eval(self, log_img_info: Dict[str, torch.Tensor]):
        B, C, H, W = log_img_info["state"].shape

        action_ij = action_xy_to_ij(log_img_info["action"], H, W)
        action_str = [
            f"action (red)\n" + 
            f"step idx {log_img_info['step_idx'][i]}\n" + 
            f"action {log_img_info['action'][i, 0]:.4f} {log_img_info['action'][i, 1]:.4f}\n"
            for i in range(B)
        ]
        qf = self._visualize_qf(self.critic, log_img_info["state"], log_img_info["step_idx"])

        return learn_utils.plot_wrap_fig(
            denses=[
                utils.torch_to_numpy(log_img_info["state"][:, 0, :, :]),
                utils.torch_to_numpy(log_img_info["state"][:, 1, :, :]),
                utils.torch_to_numpy(log_img_info["state"][:, 2, :, :]),
                utils.torch_to_numpy(log_img_info["state"][:, 3, :, :]),
                learn_utils.annotate_action(
                    denseRGB=np.repeat(utils.torch_to_numpy(log_img_info["state"][:, 0, :, :])[..., None], 3, axis=3),
                    actions=[utils.torch_to_numpy(action_ij).astype(np.int32)],
                    colors=[(1., 0., 0.)],
                    widths=[3],
                ),
                utils.torch_to_numpy(qf)
            ],
            titles=["depth", "mask_garment", "mask_inverse", "mask_hanger", action_str, "qf"],
            colorbars=["gist_gray"] * 4 + [None, "viridis"],
            plot_batch_size=log_img_info["state"].shape[0],
        )

    def log_all(self, loss_info: Dict[str, torch.Tensor], misc_info: Dict[str, torch.Tensor], name: str):
        for k, v in loss_info.items():
            self.log(f"{name}_loss/{k}", v.detach().clone().cpu())
        for k, v in misc_info.items():
            self.log(f"{name}_misc/{k}", v.detach().clone().cpu())

    def training_step(self, batch, batch_idx):
        opt_q, opt_p = self.optimizers()

        loss_q, log_img_info_q, misc_info_q = self.forward_critic(batch)
        opt_q.zero_grad()
        self.manual_backward(loss_q)
        opt_q.step()

        loss_p, log_img_info_p, misc_info_p = self.forward_policy(batch)
        opt_p.zero_grad()
        self.manual_backward(loss_p)
        opt_p.step()

        loss_info = dict(critic=loss_q, policy=loss_p)
        misc_info = {**misc_info_q, **misc_info_p}
        log_img_info = {**log_img_info_q, **log_img_info_p}
        self.log_all(loss_info, misc_info, "train")

        if (batch_idx + 1) % self.target_update_interval == 0:
            soft_update(self.critic_target, self.critic, self.tau)
    
    def validation_step(self, batch, batch_idx):
        loss_q, log_img_info_q, misc_info_q = self.forward_critic(batch)
        loss_p, log_img_info_p, misc_info_p = self.forward_policy(batch)
        
        loss_info = dict(critic=loss_q, policy=loss_p)
        misc_info = {**misc_info_q, **misc_info_p}
        log_img_info = {**log_img_info_q, **log_img_info_p}
        if batch_idx > 0:
            self.log_all(loss_info, misc_info, "valid")
        if batch_idx in self.learn_kwargs["valid"]["plot_batch_idx"]:
            self.log_img(batch_idx, log_img_info)


@dataclass
class InsertSACDataDict:
    depth_path: str
    mask_garment_path: str
    mask_inverse_path: str
    mask_hanger_path: str

    depth_next_path: str
    mask_garment_next_path: str
    mask_inverse_next_path: str
    mask_hanger_next_path: str

    reward: float
    action: np.ndarray
    done: float

    step_idx: int


class InsertSACDataset(Dataset):
    def __init__(
        self,
        data_list: List[InsertSACDataDict], 
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

        self._name = name

        print(f"insert dataset {self._name}: len {self._size}")

    def __len__(self):
        return self._size
        
    def __getitem__(self, index: int):
        # extract data
        data: InsertSACDataDict = self._data_list[self._data_index_table[index]]

        depth: np.ndarray = np.load(data.depth_path).astype(self._dtype)
        mask_garment: np.ndarray = np.load(data.mask_garment_path).astype(self._dtype)
        mask_inverse: np.ndarray = np.load(data.mask_inverse_path).astype(self._dtype)
        mask_hanger: np.ndarray = np.load(data.mask_hanger_path).astype(self._dtype)
        mask = concat_mask(dict(mask_garment=mask_garment, mask_inverse=mask_inverse, mask_hanger=mask_hanger))

        depth_next: np.ndarray = np.load(data.depth_next_path).astype(self._dtype)
        mask_garment_next: np.ndarray = np.load(data.mask_garment_next_path).astype(self._dtype)
        mask_inverse_next: np.ndarray = np.load(data.mask_inverse_next_path).astype(self._dtype)
        mask_hanger_next: np.ndarray = np.load(data.mask_hanger_next_path).astype(self._dtype)
        mask_next = concat_mask(dict(mask_garment=mask_garment_next, mask_inverse=mask_inverse_next, mask_hanger=mask_hanger_next))

        reward: float = self._dtype(data.reward)
        action: np.ndarray = data.action.astype(self._dtype)
        done: float = self._dtype(data.done)
        step_idx: int = int(data.step_idx)

        # augment data
        depth, mask, info = learn_utils.data_augmentation(depth, mask, self._ds_cfg.aug)
        depth_next, mask_next, info = learn_utils.data_augmentation(depth_next, mask_next, self._ds_cfg.aug)
        
        return dict(
           depth=depth, mask=mask, action=action, reward=reward, 
           depth_next=depth_next, mask_next=mask_next, done=done, step_idx=step_idx,
        )


def make_insert_dataset(
    data_path: List[str], 
    valid_size_raw: float,
    ds_cfg: omegaconf.DictConfig,
):
    pattern = "^completed.txt$"
    df = learn_utils.DataForest(data_path, [pattern])
    node_n_raw = df.get_forest_size(pattern)
    
    data_list = []
    print("scanning all trajectories ...")
    for idx in tqdm.tqdm(range(node_n_raw)):
        misc_path = os.path.join(os.path.dirname(df.get_item(pattern, idx).file_path), "misc.json")
        with open(misc_path, "r") as f_obj:
            misc_info = json.load(f_obj)
        base_dir = os.path.dirname(misc_path)

        def format_batch(batch_idx):
            return str(batch_idx).zfill(len(str(misc_info["batch_size"] - 1)))

        def calculate_reward(score, sim_error, sim_error_next, batch_idx, step_idx):
            if not sim_error["all"][batch_idx] and sim_error_next["all"][batch_idx]:
                return -1
            if step_idx == 1:
                return int(score["left"][batch_idx] >= REWARD_THRESHOLD["left"])
            elif step_idx == 3:
                return int(score["right"][batch_idx] >= REWARD_THRESHOLD["right"])
            else:
                return 0

        with open(os.path.join(base_dir, "score", "2.json"), "r") as f_obj:
            score_batch_2 = json.load(f_obj)
        
        for step_idx in range(4):
            action_xy_batch = np.load(os.path.join(base_dir, "action", f"{step_idx}.npy"), allow_pickle=True).item()["xy"]
            with open(os.path.join(base_dir, "score", f"{step_idx+1}.json"), "r") as f_obj:
                score_batch = json.load(f_obj)
            with open(os.path.join(base_dir, "sim_error", f"{step_idx}.json"), "r") as f_obj:
                sim_error_batch = json.load(f_obj)
            with open(os.path.join(base_dir, "sim_error", f"{step_idx+1}.json"), "r") as f_obj:
                sim_error_batch_next = json.load(f_obj)
            
            for batch_idx in range(int(misc_info["batch_size"])):
                if sim_error_batch["all"][batch_idx]: # skip sim_error batch
                    continue
                if step_idx in [2, 3] and score_batch_2["left"][batch_idx] < REWARD_THRESHOLD["left"]: # skip step 2, 3 if step 0, 1 are fail
                    continue
                obs_base_dir = os.path.join(base_dir, "obs", format_batch(batch_idx))

                # assemble data
                data_dict = InsertSACDataDict(
                    depth_path=os.path.join(obs_base_dir, "reproject_depth", f"{step_idx}.npy"),
                    mask_garment_path=os.path.join(obs_base_dir, "reproject_is_garment_nointerp", f"{step_idx}.npy"),
                    mask_inverse_path=os.path.join(obs_base_dir, "reproject_is_inverse_nointerp", f"{step_idx}.npy"),
                    mask_hanger_path=os.path.join(obs_base_dir, "reproject_is_hanger_nointerp", f"{step_idx}.npy"),

                    depth_next_path=os.path.join(obs_base_dir, "reproject_depth", f"{step_idx+1}.npy"),
                    mask_garment_next_path=os.path.join(obs_base_dir, "reproject_is_garment_nointerp", f"{step_idx+1}.npy"),
                    mask_inverse_next_path=os.path.join(obs_base_dir, "reproject_is_inverse_nointerp", f"{step_idx+1}.npy"),
                    mask_hanger_next_path=os.path.join(obs_base_dir, "reproject_is_hanger_nointerp", f"{step_idx+1}.npy"),

                    reward=calculate_reward(score_batch, sim_error_batch, sim_error_batch_next, batch_idx, step_idx),
                    action=action_xy_batch[batch_idx, :],
                    done=float(step_idx == 3),

                    step_idx=int(step_idx),
                )
                data_list.append(data_dict)

    total_size = len(data_list)
    valid_size = learn_utils.parse_size(valid_size_raw, total_size)
    path_idx_permutated = np.random.permutation(total_size)
    trds = InsertSACDataset(data_list, path_idx_permutated[valid_size:], ds_cfg, name=f"train")
    vlds = InsertSACDataset(data_list, path_idx_permutated[:valid_size], ds_cfg, name=f"valid")
    return trds, vlds


class InsertPolicySAC(InsertPolicy):
    def __init__(self, insert_gym, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)

        self._module = SACModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()

    def _net_inference(
        self, 
        step_idx: int,
        info: dict,
    ):  
        # input
        obs_all = dict(depth=[], mask=[])

        # render and store result
        for batch_idx in range(self.batch_size):
            result, mask_str_to_idx, camera_info = self._agent.get_obs("direct", "small", batch_idx, True, pos=self._agent.direct_obs)
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=False, target="double_side")
            depth_raw = reproject_result["depth_output"]
            mask_garment = reproject_result["mask_output"]
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=False, target="inverse_side")
            mask_inverse = reproject_result["mask_output"]
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=False, target="hanger")
            mask_hanger = reproject_result["mask_output"]

            mask_raw = concat_mask(dict(mask_garment=mask_garment, mask_inverse=mask_inverse, mask_hanger=mask_hanger))

            obs_all["depth"].append(depth_raw)
            obs_all["mask"].append(mask_raw)
        
        # inference
        depth = torch.tensor(np.array(obs_all["depth"]), device=self.device, dtype=self.dtype) # [B, H, W]
        B, H, W = depth.shape

        mask = torch.tensor(np.array(obs_all["mask"]), device=self.device, dtype=self.dtype) # [B, 3, H, W]
        assert mask.shape == (B, 3, H, W), mask.shape

        with torch.no_grad():
            action_xy, log_img_info = self._module.run_predict(depth, mask, torch.tensor([step_idx] * B, dtype=torch.long, device=self.device))
        ret_action = dict(action_xy=action_xy.to(device=self.device).detach())
        ret_info = dict()
        if "log_path" in info.keys():
            for batch_idx in range(B):
                output_path = os.path.join(info["log_path"], str(batch_idx).zfill(len(str(B-1))) + ".pdf")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self._module.log_img_eval({
                    k: v[[batch_idx], ...] for k, v in log_img_info.items()
                }).savefig(output_path)
                plt.close()

        return dict(action=ret_action, info=ret_info)
    
    def get_press_action(self, info: dict):
        action_and_info = self._net_inference(0, info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_xy"])
        return dict(action=action, info=action_and_info["info"])
    
    def get_lift_action(self, info: dict):
        action_and_info = self._net_inference(1, info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_xy"])
        return dict(action=action, info=action_and_info["info"])
    
    def get_drag_action(self, info: dict):
        action_and_info = self._net_inference(2, info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_xy"])
        return dict(action=action, info=action_and_info["info"])
    
    def get_rotate_action(self, info: dict):
        action_and_info = self._net_inference(3, info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_xy"])
        return dict(action=action, info=action_and_info["info"])