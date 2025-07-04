import logging
logger = logging.getLogger(__name__)

import copy
from typing import Dict, Literal, Any, List, Callable, Tuple
import time
import math
import os

import cv2
import tqdm

import torch
import numpy as np
import matplotlib.pyplot as plt

import omegaconf

import robohang.common.utils as utils
import robohang.sim.sim_utils as sim_utils
import robohang.policy.policy_utils as policy_utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
from robohang.policy.funnel.funnel_gym import FunnelGym
from robohang.policy.funnel.funnel_learn import (
    FunnelFlingNormalModule,
    FunnelActEncDec,
)


class TimerFunnelPolicy:
    @staticmethod
    def timer(func: Callable):
        def wrapper(*args, **kwargs):
            tic = time.time()
            result = func(*args, **kwargs)
            toc = time.time()
            logger.info(f"TimerFunnelPolicy: {func} costs {toc - tic} s")
            return result
        return wrapper


class FunnelPolicy:
    def __init__(self, funnel_gym: FunnelGym, policy_cfg: omegaconf.DictConfig) -> None:
        self._policy_cfg = copy.deepcopy(policy_cfg)
        self._funnel_gym = funnel_gym
        self._agent = funnel_gym.agent

        self.dtype = self._funnel_gym.dtype
        self.dtype_int = self._funnel_gym.dtype_int
        self.device = self._funnel_gym.device
        self.batch_size = self._funnel_gym.batch_size

        self._B_idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

    def get_fling_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._funnel_gym.random_fling(),
            info=dict(),
        )
    
    def get_pick_place_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._funnel_gym.random_pick_place(),
            info=dict(),
        )


class FunnelPolicyRandom(FunnelPolicy):
    def __init__(self, funnel_gym: FunnelGym, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(funnel_gym, policy_cfg)

    def get_fling_action(self) -> Dict[Literal["action", "info"], Any]:
        '''print("[WARN] debug", __name__)
        return dict(
            action=dict(
                center_xy=torch.tensor([[-0.10, 0.65], [-0.10, 0.65], [+0.10, 0.35], [+0.10, 0.35]], dtype=self.dtype, device=self.device),
                distance=torch.tensor([0.25, 0.40, 0.25, 0.40], dtype=self.dtype, device=self.device),
                angle_degree=torch.tensor([-45., -45., -45., -45.], dtype=self.dtype, device=self.device),
            ),
            info=dict(),
        )'''
        action_cfg = self._policy_cfg.fling
        per_batch = action_cfg.query.per_batch
        num_iter = action_cfg.query.num_iter
        dx_eps = self._funnel_gym.garment.dx_eps

        idx_dict = dict(fail=0, normal=1)
        target_B1 = torch.tensor(
            np.random.choice(
                a=2, size=(self.batch_size, ), 
                p=[action_cfg.prob.fail, action_cfg.prob.normal]
            ), dtype=self.dtype_int, device=self.device,
        )[:, None]

        def pack_sample(x: Dict[str, torch.Tensor]):
            """x: Dict[str -> [B, D]]"""
            return torch.concat([
                x["center_xy"],
                x["distance"][:, None],
                x["angle_degree"][:, None],
            ], dim=1).view(self.batch_size, per_batch, 4)
        
        def unpack_sample(x: torch.Tensor):
            return dict(
                center_xy=x[..., :2],
                distance=x[..., 2],
                angle_degree=x[..., 3],
            )
        
        def generate_sample_func():
            return pack_sample(self._funnel_gym.random_fling(self.batch_size * per_batch))
        
        def is_good_sample_func(sample: torch.Tensor):
            """sample: [B, D, 5]"""
            unpacked = unpack_sample(sample)
            center = unpacked["center_xy"]
            angle_rad = torch.deg2rad(unpacked["angle_degree"])
            distance = unpacked["distance"]

            query_xy_l = center.clone()
            query_xy_l[..., 0] -= distance / 2. * torch.cos(angle_rad)
            query_xy_l[..., 1] -= distance / 2. * torch.sin(angle_rad)

            query_xy_r = center.clone()
            query_xy_r[..., 0] += distance / 2. * torch.cos(angle_rad)
            query_xy_r[..., 1] += distance / 2. * torch.sin(angle_rad)

            pos = self._funnel_gym.garment.get_pos()
            f2v = self._funnel_gym.garment.get_f2v()

            satisfy_rule_l = torch.zeros((self.batch_size, per_batch), dtype=self.dtype_int, device=self.device)
            satisfy_rule_r = torch.zeros((self.batch_size, per_batch), dtype=self.dtype_int, device=self.device)
            for rule in action_cfg.heuristic:
                if len(rule) == 4:
                    x, y, f, n = rule
                    assert f in ["eq", "ge", "le"], f
                    dxy = torch.tensor([[x, y]], dtype=self.dtype, device=self.device)
                    layer_num_l = policy_utils.calculate_garment_layer(pos, f2v, query_xy_l + dxy, dx_eps)["layer_num"] # [B, Q]
                    layer_num_r = policy_utils.calculate_garment_layer(pos, f2v, query_xy_r + dxy, dx_eps)["layer_num"] # [B, Q]
                    satisfy_rule_l = torch.logical_or(satisfy_rule_l, getattr(torch, f)(layer_num_l, n))
                    satisfy_rule_r = torch.logical_or(satisfy_rule_r, getattr(torch, f)(layer_num_r, n))
                else:
                    raise ValueError(rule)

            layer_num_l = policy_utils.calculate_garment_layer(pos, f2v, query_xy_l, dx_eps)["layer_num"] # [B, Q]
            layer_num_r = policy_utils.calculate_garment_layer(pos, f2v, query_xy_r, dx_eps)["layer_num"] # [B, Q]
            is_success = torch.logical_and(
                torch.logical_and(satisfy_rule_l, satisfy_rule_r),
                torch.logical_and(layer_num_l >= 1, layer_num_r >= 1)
            )
            is_good_sample = torch.logical_or(
                torch.logical_and(target_B1 == idx_dict["fail"], torch.logical_not(is_success)), # fail
                torch.logical_and(target_B1 == idx_dict["normal"], is_success),
            )
            return is_good_sample.to(dtype=self.dtype_int)

        current_answer, current_is_good = policy_utils.iterative_sample(
            generate_sample_func,
            is_good_sample_func,
            num_iter,
        )
        info = dict(
            target=target_B1[:, 0],
            is_good=current_is_good,
            idx_dict=idx_dict,
        )
        logger.info(f"get_pick_place_action:{info}")
        return dict(action=unpack_sample(current_answer), info=info)

    def get_pick_place_action(self) -> Dict[Literal["action", "info"], Any]:
        action_cfg = self._policy_cfg.pick_place
        per_batch = action_cfg.query.per_batch
        num_iter = action_cfg.query.num_iter
        dx_eps = self._funnel_gym.garment.dx_eps

        idx_dict = dict(fail=0, success=1)
        target_B1 = torch.tensor(
            np.random.choice(
                a=2, size=(self.batch_size, ), 
                p=[action_cfg.prob.fail, action_cfg.prob.success]
            ), dtype=self.dtype_int, device=self.device,
        )[:, None]

        def pack_sample(x: Dict[str, torch.Tensor]):
            """x: Dict[str -> [B, D]]"""
            return torch.concat([
                x["xy_s"],
                x["xy_e"]
            ], dim=1).view(self.batch_size, per_batch, 4)
        
        def unpack_sample(x: torch.Tensor):
            return dict(
                xy_s=x[..., :2],
                xy_e=x[..., 2:],
            )
        
        def generate_sample_func():
            return pack_sample(self._funnel_gym.random_pick_place(self.batch_size * per_batch))
        
        def is_good_sample_func(sample: torch.Tensor):
            """sample: [B, D, 5]"""
            unpacked = unpack_sample(sample)
            xy_s = unpacked["xy_s"]

            pos = self._funnel_gym.garment.get_pos()
            f2v = self._funnel_gym.garment.get_f2v()
            
            layer_num = policy_utils.calculate_garment_layer(pos, f2v, xy_s, dx_eps)["layer_num"] # [B, Q]
            logger.info(f"query_xy_l:{layer_num} layer_num_l:{xy_s} ")

            is_fail = (layer_num == 0)
            is_good_sample = torch.logical_or(
                torch.logical_and(target_B1 == idx_dict["fail"], is_fail), # fail
                torch.logical_and(target_B1 == idx_dict["success"], torch.logical_not(is_fail)), # success
            )
            return is_good_sample.to(dtype=self.dtype_int)

        current_answer, current_is_good = policy_utils.iterative_sample(
            generate_sample_func,
            is_good_sample_func,
            num_iter,
        )
        info = dict(
            target=target_B1[:, 0],
            is_good=current_is_good,
            idx_dict=idx_dict,
        )
        logger.info(f"get_pick_place_action:{info}")
        return dict(action=unpack_sample(current_answer), info=info)


def prepare_obs(
    depth_raw: np.ndarray,
    mask_raw: np.ndarray,
    angle_degree_list: List[float],
    distance_list: List[float],
    distance_range: List[float],
):
    action_space_raw = np.zeros_like(depth_raw)
    i, j = MetaInfo.get_action_space_slice("fn", depth_raw)
    action_space_raw[i, j] = 1.

    obs_list = dict(depth=[], mask=[], action_space=[], depth_raw=[], mask_raw=[])
    angle_degree_sample = []
    distance_sample = []
    for a in angle_degree_list:
        for d in distance_list:
            result = FunnelActEncDec.transform_image_and_action(
                distance_range=distance_range,
                depth=depth_raw, 
                mask=mask_raw, 
                action_space=action_space_raw, 
                action_angle_degree=a,
                action_distance=d,
            )
            # store result
            for k in obs_list.keys():
                if k == "depth_raw":
                    obs_list[k].append(depth_raw)
                elif k == "mask_raw":
                    obs_list[k].append(mask_raw)
                else:
                    obs_list[k].append(result[k])
            angle_degree_sample.append(a)
            distance_sample.append(d)
    
    return obs_list, angle_degree_sample, distance_sample


def calculate_action_direct_ij(
    action: Dict[str, torch.Tensor], 
    depth: torch.Tensor, 
    dtype: torch.dtype,
    device: torch.device
):
    """action: [B, 4]"""
    B, H, W = depth.shape
    depth_np = utils.torch_to_numpy(depth)
    action_np = utils.torch_dict_to_numpy_dict(action)

    action_direct_ij = []
    for batch_idx in range(B):
        z = MetaInfo.calculate_z_world(depth_np[batch_idx, ...])
        x, y = action_np["center_xy"][batch_idx]
        dist = action_np["distance"][batch_idx]
        ang_rad = math.radians(action_np["angle_degree"][batch_idx])

        dij = np.zeros(4, dtype=depth_np.dtype)
        dij[FunnelActEncDec.fling_direct_ij_left_idx] = policy_utils.xyz2ij(
            x - dist * math.cos(ang_rad) / 2,
            y - dist * math.sin(ang_rad) / 2,
            z, MetaInfo.reproject_camera_info
        )
        dij[FunnelActEncDec.fling_direct_ij_right_idx] = policy_utils.xyz2ij(
            x + dist * math.cos(ang_rad) / 2,
            y + dist * math.sin(ang_rad) / 2,
            z, MetaInfo.reproject_camera_info
        )
        action_direct_ij.append(dij)
    
    return torch.tensor(np.array(action_direct_ij), dtype=dtype, device=device)


def inference_fling(
    fn_module: FunnelFlingNormalModule,
    obs_all: Dict[str, List[np.ndarray]],
    angle_degree_all: List[List[float]],
    distance_all: List[List[float]],
    distance_range: List[float],
    device: torch.device,
    dtype: torch.dtype,
    info: dict,
) -> Tuple[Dict[Literal["center_xy", "distance", "angle_degree"], torch.Tensor], dict]:
    device_data = "cpu"

    # extract data
    depth = torch.tensor(np.array(obs_all["depth"]), device=device_data, dtype=dtype) # [B, S, H, W]
    B, S, H, W = depth.shape
    B_idx = torch.arange(B, dtype=torch.long, device=device_data)

    mask = torch.tensor(np.array(obs_all["mask"]), device=device_data, dtype=dtype) # [B, S, H, W]
    assert mask.shape == (B, S, H, W), mask.shape

    action_space = torch.tensor(np.array(obs_all["action_space"]), device=device_data, dtype=dtype) # [B, S, H, W]
    assert action_space.shape == (B, S, H, W), action_space.shape

    depth_raw = torch.tensor(np.array(obs_all["depth_raw"]), device=device_data, dtype=dtype) # [B, S, H, W]
    assert depth_raw.shape == (B, S, H, W), depth_raw.shape

    mask_raw = torch.tensor(np.array(obs_all["mask_raw"]), device=device_data, dtype=dtype) # [B, S, H, W]
    assert mask_raw.shape == (B, S, H, W), mask_raw.shape

    # inference
    i = fn_module.preprocess_input(depth.view(B*S, H, W), mask.view(B*S, H, W))
    dense = []
    for bs_idx in range(B*S):
        dense.append(fn_module.to(device=device).run_predict(
            i=i[[bs_idx], ...].to(device=device)
        ).detach().clone().to(device=device_data))
    dense = torch.concat(dense, dim=0)

    # select best
    coverage_qvalue_pred: torch.Tensor = dense # [B*S, H, W]
    best_indices = learn_utils.out_of_action_space_to_min(
        coverage_qvalue_pred, action_space.view(B*S, H, W)
    ).view(B, -1).max(dim=-1).indices.to(dtype=torch.long) # [B, ]
    S_idx = best_indices // (H * W) # [B, ]
    I_idx = best_indices % (H * W) // W # [B, ]
    J_idx = best_indices % W # [B, ]

    action_angle_degree = torch.tensor(
        angle_degree_all, dtype=dtype, device=device_data
    )[B_idx, S_idx] # [B, ]
    action_distance = torch.tensor(
        distance_all, dtype=dtype, device=device_data
    )[B_idx, S_idx] # [B, ]
    action_center_ij_rotated = torch.concat([I_idx[:, None], J_idx[:, None]], dim=1) # [B, 2]

    # rotate ij IMPORTANT
    action_center_ij = action_center_ij_rotated.clone()
    for batch_idx in range(B):
        center = (W // 2, H // 2)
        angle = float(action_angle_degree[batch_idx])
        scale = float(action_distance[batch_idx]) / min(*distance_range)
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
        i_u, j_u = utils.torch_to_numpy(action_center_ij_rotated[batch_idx])
        j, i = rotation_matrix @ np.array([j_u, i_u, 1.], dtype=float)
        action_center_ij[batch_idx, :] = torch.tensor([i, j])

    action_center_xy = policy_utils.action_batch_ij_to_xy(
        action_center_ij, 
        depth_raw[B_idx, S_idx, :, :]
    ) # [B, 2]

    ret_action = dict(
        center_xy=action_center_xy.to(device=device),
        distance=action_distance.to(device=device),
        angle_degree=action_angle_degree.to(device=device),
    )
    action_direct_ij_raw = calculate_action_direct_ij(ret_action, depth_raw[B_idx, S_idx, :, :], dtype, device_data) # [B, 4]
    ret_info = dict(action_direct_ij_raw=action_direct_ij_raw)

    if "log_path" in info.keys():
        action_raw = torch.zeros((B, 4), device=device_data, dtype=dtype)
        action_raw[:, FunnelActEncDec.fling_angle_degree_idx] = action_angle_degree
        action_raw[:, FunnelActEncDec.fling_distance_idx] = action_distance
        action_raw[:, FunnelActEncDec.fling_center_ij_idx] = action_center_ij
        for batch_idx in range(B):
            output_path = os.path.join(info["log_path"], str(batch_idx).zfill(len(str(B-1))) + ".pdf")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            s_idx = S_idx[batch_idx]
            dense_info = dict(
                dense=dense.view(B, S, H, W)[[batch_idx], s_idx, ...],
                action_space=action_space[[batch_idx], s_idx, ...],
                depth=depth[[batch_idx], s_idx, ...],
                mask=mask[[batch_idx], s_idx, ...],
                depth_raw=depth_raw[[batch_idx], s_idx, ...],
                mask_raw=mask_raw[[batch_idx], s_idx, ...],
                action_raw=action_raw[[batch_idx], ...],
                action_direct_ij_raw=action_direct_ij_raw[[batch_idx], ...],
                action_center_ij_rotated=action_center_ij_rotated[[batch_idx], ...]
            )
            fn_module.log_img_eval(dense_info).savefig(output_path)
            plt.close()
    
    return ret_action, ret_info


class FunnelPolicyUNet(FunnelPolicy):
    def __init__(self, funnel_gym: FunnelGym, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(funnel_gym, policy_cfg)

        self._fn_module = FunnelFlingNormalModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()

    def get_angle_degree_list(self) -> List[float]:
        as_cfg = self._funnel_gym.fling_normal_action_space
        return np.linspace(
            min(as_cfg.angle_degree),
            max(as_cfg.angle_degree),
            self._policy_cfg.angle_degree_num,
        ).tolist()

    def get_distance_list(self) -> List[float]:
        as_cfg = self._funnel_gym.fling_normal_action_space
        return np.linspace(
            min(as_cfg.distance),
            max(as_cfg.distance),
            self._policy_cfg.distance_num,
        ).tolist()
    
    def distance_range(self) -> List[float]:
        return omegaconf.OmegaConf.to_container(self._funnel_gym.fling_normal_action_space.distance)

    @TimerFunnelPolicy.timer
    def get_fling_action(self, info: dict) -> Dict[Literal['action', 'info'], Any]:
        # obs input
        obs_all = dict(depth=[], mask=[], action_space=[], depth_raw=[], mask_raw=[])

        # action
        angle_degree_all = []
        distance_all = []

        # render and store result
        for batch_idx in range(self.batch_size):
            result, mask_str_to_idx, camera_info = self._agent.get_obs("direct", "small", batch_idx, True, pos=self._agent.direct_obs)
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=self._policy_cfg.interp_mask, target="double_side")
            
            depth_raw = reproject_result["depth_output"]
            mask_raw = reproject_result["mask_output"]
            
            obs_list, angle_degree_sample, distance_sample = prepare_obs(
                depth_raw, 
                mask_raw, 
                self.get_angle_degree_list(), 
                self.get_distance_list(), 
                self.distance_range()
            )
            # store result
            for k in obs_all.keys():
                obs_all[k].append(obs_list[k])
            angle_degree_all.append(angle_degree_sample) # [B, S]
            distance_all.append(distance_sample) # [B, S]
        
        ret_action, ret_info = inference_fling(
            self._fn_module,
            obs_all, 
            angle_degree_all,
            distance_all,
            self.distance_range(),
            self.device,
            self.dtype,
            info,
        )
        
        return dict(action=ret_action, info=ret_info)