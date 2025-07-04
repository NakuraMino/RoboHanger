import logging
logger = logging.getLogger(__name__)

import taichi as ti

import copy
from typing import List, Dict, Literal, Any, Tuple, Union
import os

import torch
import numpy as np
import tqdm
import cv2
import matplotlib.pyplot as plt

import omegaconf

import robohang.common.utils as utils
import robohang.policy.policy_utils as policy_utils
import robohang.policy.learn_utils as learn_utils
import robohang.policy.funnel.funnel_policy as funnel_policy_module
from robohang.policy.funnel.funnel_learn import FunnelFlingNormalModule
import robohang.policy.insert.insert_policy as insert_policy_module
from robohang.policy.insert.insert_learn import InsertLeftEndModule, InsertRightEndModule, DualArmModule, concat_mask
from robohang.policy.insert.insert_learn_imi import InsertLeftEndImiModule, InsertRightEndImiModule, DualArmImiModule
from robohang.policy.funnel.keypoints_unet import KeypointsModule
from robohang.policy.insert.insert_learn_act import InsertACTModule, MAX_DEPTH


@ti.kernel
def _depth_to_point_cloud_kernel(
    depth: ti.types.ndarray(),
    pc: ti.types.ndarray(dtype=ti.math.vec3),
    extrinsics: ti.math.mat4,
    intrinsics: ti.math.mat3,
):
    H, W = depth.shape
    model_matrix = ti.Matrix(policy_utils.model_matrix_np, dt=float)
    for i, j in ti.ndrange(H, W):
        k = i * W + j
        uvw = ti.Vector([j + 0.5, i + 0.5, 1.], dt=float)
        x, y, z = (ti.math.inverse(intrinsics) @ uvw) * depth[i, j]
        xyzw = extrinsics @ model_matrix @ ti.Vector([x, y, z, 1.], dt=float)
        pc[k] = xyzw[:3]


def depth_to_point_cloud(
    depth: np.ndarray,
    extrinsics: np.ndarray,
    intrinsics: np.ndarray,
    dtype=np.float32,
):
    H, W = depth.shape
    pc = np.zeros((H * W, 3), dtype=dtype)
    _depth_to_point_cloud_kernel(depth, pc, extrinsics, intrinsics)
    return pc


def sample_point_cloud(xyz: np.ndarray, num: int):
    """
    xyz: [N, 3]
    ```
    if N <= num:
        return xyz.copy()
    else:
        return num xyzs from xyz
    ```
    """
    N, D = xyz.shape
    assert D == 3, xyz.shape
    if N <= num:
        return xyz.copy()
    else:
        return xyz[np.random.permutation(N)[:num], :]


def align_point_cloud(
    xyz_real: torch.Tensor,
    xyz_gt: torch.Tensor,
    device="cuda",
    dtype=torch.float32,
    sample_batch=32,
    iter_n=64,
    lr=1e-1,
    beta=0.9,
    use_tqdm=True,
):
    """
    Find `mat`, which minimize the chamfer between `mat @ xyz_gt` and `xyz_real`
    """
    N, D = xyz_real.shape
    assert D == 3, xyz_real.shape
    M, D = xyz_gt.shape
    assert D == 3, xyz_gt.shape
    S = int(sample_batch)

    # init
    xya_np = np.zeros((S, 3))
    xya_np[:, 2] = np.linspace(0., np.pi * 2, S, endpoint=False)
    xya_np[:, 0:2] = utils.torch_to_numpy(xyz_real.mean(dim=0) - xyz_gt.mean(dim=0))[:2]
    xya = torch.tensor(xya_np, dtype=dtype, device=device, requires_grad=False) # [S, 3]
    xyz_gt = xyz_gt.to(dtype=dtype, device=device).requires_grad_(False)
    xyz_real = xyz_real.to(dtype=dtype, device=device).requires_grad_(False)
    momentum = torch.zeros_like(xya, requires_grad=False)

    # iterative optimize
    for _ in (tqdm.tqdm(range(iter_n)) if use_tqdm else range(iter_n)):
        xya.requires_grad_(True)
        x = xya[:, 0, None] + xyz_gt[None, :, 0] * torch.cos(xya[:, 2, None]) - xyz_gt[None, :, 1] * torch.sin(xya[:, 2, None]) # [S, M]
        y = xya[:, 1, None] + xyz_gt[None, :, 1] * torch.cos(xya[:, 2, None]) + xyz_gt[None, :, 0] * torch.sin(xya[:, 2, None]) # [S, M]
        dist_sqr = (x[:, :, None] - xyz_real[None, None, :, 0]) ** 2 + (y[:, :, None] - xyz_real[None, None, :, 1]) ** 2 # [S, M, N]
        loss: torch.Tensor = dist_sqr.min(dim=1).values.mean(dim=1) # [S]
        torch.sum(loss).backward()
        momentum = beta * momentum + xya.grad
        xya = (xya - lr * momentum).detach()
    
    # get result
    best_xya = utils.torch_to_numpy(xya[torch.argmin(loss)])
    mat = np.eye(4)
    mat[0:2, 3] = best_xya[0:2]
    mat[0, 0] = mat[1, 1] = np.cos(best_xya[2])
    mat[0, 1] = -np.sin(best_xya[2])
    mat[1, 0] = +np.sin(best_xya[2])
    return mat


def inference_insert(
    module: Union[DualArmModule, DualArmImiModule],
    obs_all: Dict[str, List[np.ndarray]],
    device: torch.device,
    dtype: torch.dtype,
    info: dict,
) -> Tuple[Dict[Literal["action_1_xy", "action_2_xy"], torch.Tensor], Dict[Literal["action_1_z"], torch.Tensor]]:
    # extract data
    depth = torch.tensor(np.array(obs_all["depth"]), device=device, dtype=dtype) # [B, H, W]
    B, H, W = depth.shape

    mask = torch.tensor(np.array(obs_all["mask"]), device=device, dtype=dtype) # [B, 3, H, W]
    assert mask.shape == (B, 3, H, W), mask.shape

    action_1_space = torch.tensor(np.array(obs_all["action_1_space"]), device=device, dtype=dtype) # [B, H, W]
    assert action_1_space.shape == (B, H, W), action_1_space.shape

    action_2_space = torch.tensor(np.array(obs_all["action_2_space"]), device=device, dtype=dtype) # [B, H, W]
    assert action_2_space.shape == (B, H, W), action_2_space.shape

    # inference
    action_1_xy, action_2_xy, dense_info = module.run_predict(depth, mask, action_1_space, action_2_space)
    ret_action = dict(action_1_xy=action_1_xy, action_2_xy=action_2_xy)
    ret_info = dict()
    if "log_path" in info.keys():
        for batch_idx in range(B):
            output_path = os.path.join(info["log_path"], str(batch_idx).zfill(len(str(B-1))) + ".pdf")
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            module.log_img_eval({
                k: v[[batch_idx], ...] for k, v in dense_info.items()
            }).savefig(output_path)
            plt.close()
            
            np.save(
                os.path.join(info["log_path"], "0.npy"), 
                utils.torch_dict_to_numpy_dict(dict(depth=depth, mask=mask, dense_info=dense_info)), 
                allow_pickle=True
            )

    action_1_ij = dense_info["action_1_ij"]
    action_1_z = policy_utils.action_batch_ij_to_z(action_1_ij, depth)
    ret_info["action_1_z"] = action_1_z

    return ret_action, ret_info


class RealPolicy:
    def __init__(self, cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        self._cfg = copy.deepcopy(cfg)

        # factory args
        self._dtype: torch.dtype = getattr(torch, global_cfg.default_float)
        assert isinstance(self._dtype, torch.dtype)
        self._dtype_int: torch.dtype = getattr(torch, global_cfg.default_int)
        assert isinstance(self._dtype_int, torch.dtype)
        self._device: str = str(global_cfg.torch_device)

        try:
            # funnel module
            self._fn_module = FunnelFlingNormalModule.load_from_checkpoint(
                utils.get_path_handler()(self._cfg.funnel.ckpt), 
                map_location=torch.device("cpu"),
            ).to(self._device).eval()

            # keypoint module
            self._kp_module = KeypointsModule.load_from_checkpoint(
                utils.get_path_handler()(self._cfg.keypoint.ckpt), 
                map_location=torch.device("cpu"),
            ).to(self._device).eval()
        except Exception as e:
            print(e)
            print("\ninit real policy failed ...\n")
        
        self._policy_type = str(self._cfg.insert.type)
        if self._policy_type == "qf":
            try:
                # insert module
                self._le_module = InsertLeftEndModule.load_from_checkpoint(
                    utils.get_path_handler()(self._cfg.insert.ckpt.left), 
                    map_location=torch.device("cpu"),
                ).to(self._device).eval()

                self._re_module = InsertRightEndModule.load_from_checkpoint(
                    utils.get_path_handler()(self._cfg.insert.ckpt.right), 
                    map_location=torch.device("cpu"),
                ).to(self._device).eval()
            except Exception as e:
                print(e)
                print("\ninit real policy failed ...\n")
        elif self._policy_type == "imi":
            try:
                # insert module
                self._le_module = InsertLeftEndImiModule.load_from_checkpoint(
                    utils.get_path_handler()(self._cfg.insert.ckpt.left), 
                    map_location=torch.device("cpu"),
                ).to(self._device).eval()

                self._re_module = InsertRightEndImiModule.load_from_checkpoint(
                    utils.get_path_handler()(self._cfg.insert.ckpt.right), 
                    map_location=torch.device("cpu"),
                ).to(self._device).eval()
            except Exception as e:
                print(e)
                print("\ninit real policy failed ...\n")
        elif self._policy_type == "fix":
            self._policy_output = self._cfg.insert.value
        elif self._policy_type == "act":
            self._module = InsertACTModule.load_from_checkpoint(
                utils.get_path_handler()(self._cfg.insert.ckpt), 
                map_location=torch.device("cpu"),
            ).to(self._device).eval()
            if self._cfg.insert.action_w == "default":
                raise NotImplementedError
                self.action_w = 1.0 / self._module.actpred_len * (-1.) # different from origin paper which action_w is positive
            else:
                self.action_w = float(self._cfg.insert.action_w)
            self._obs_cache = []
            self._sta_cache = []
            self._act_cache = []
        else:
            raise NotImplementedError(self._policy_type)

    def set_funnel_action_space(self, action_space_cfg: omegaconf.DictConfig):
        self._funnel_action_space = copy.deepcopy(action_space_cfg)
    
    def _tensor(self, data):
        return torch.tensor(np.array(data), dtype=self._dtype, device=self._device)
    
    def _get_funnel_angle_degree_list(self):
        as_cfg = self._funnel_action_space
        return np.linspace(
            min(as_cfg.angle_degree),
            max(as_cfg.angle_degree),
            self._cfg.funnel.angle_degree_num,
        ).tolist()

    def _get_funnel_distance_list(self):
        as_cfg = self._funnel_action_space
        return np.linspace(
            min(as_cfg.distance),
            max(as_cfg.distance),
            self._cfg.funnel.distance_num,
        ).tolist()

    def _get_funnel_distance_range(self):
        as_cfg = self._funnel_action_space
        return np.linspace(
            min(as_cfg.distance),
            max(as_cfg.distance),
            self._cfg.funnel.distance_num,
        ).tolist()

    def _sim_z_to_real_z(self, sim_z: torch.Tensor, real_reproject_camera_z: float):
        return float(sim_z - policy_utils.MetaInfo.reproject_camera_info.extri[2, 3] + real_reproject_camera_z)
    
    def _check_depth_raw_shape(self, depth_raw: np.ndarray):
        assert depth_raw.shape == (128, 128)

    def predict_fling(
        self, 
        depth_raw: np.ndarray, 
        mask_raw: np.ndarray, 
        info: dict
    ):
        self._check_depth_raw_shape(depth_raw)

        # obs input
        obs_all = dict(depth=[], mask=[], action_space=[], depth_raw=[], mask_raw=[])

        # action
        angle_degree_all = []
        distance_all = []

        for batch_idx in range(1):
            obs_list, angle_degree_sample, distance_sample = funnel_policy_module.prepare_obs(
                depth_raw, 
                mask_raw, 
                self._get_funnel_angle_degree_list(), 
                self._get_funnel_distance_list(), 
                self._get_funnel_distance_range()
            )
            # store result
            for k in obs_all.keys():
                obs_all[k].append(obs_list[k])
            angle_degree_all.append(angle_degree_sample) # [B, S]
            distance_all.append(distance_sample) # [B, S]

        ret_action, ret_info = funnel_policy_module.inference_fling(
            self._fn_module,
            obs_all, 
            angle_degree_all,
            distance_all,
            self._get_funnel_distance_range(),
            self._device,
            self._dtype,
            info,
        )

        return dict(action=ret_action, info=ret_info)

    def predict_insert(
        self, 
        depth_raw: np.ndarray, 
        mask_garment: np.ndarray, 
        mask_hanger: np.ndarray, 
        mask_neckline: np.ndarray,
        real_reproject_camera_z: float,
        endpoint_name: Literal["left", "right"],
        info: dict,
    ):
        assert self._policy_type in ["qf", "imi", "fix"]

        if self._policy_type == "fix":
            action_1_xy = self._tensor([getattr(self._policy_output, endpoint_name).action_1])
            action_2_xy = self._tensor([getattr(self._policy_output, endpoint_name).action_2])
            return dict(action=dict(action_1_xy=action_1_xy, action_2_xy=action_2_xy), info=dict())
        
        self._check_depth_raw_shape(depth_raw)

        if endpoint_name == "left":
            action_1_str = "press"
            action_2_str = "lift"
            module = self._le_module
        elif endpoint_name == "right":
            action_1_str = "drag"
            action_2_str = "rotate"
            module = self._re_module
        else:
            raise ValueError(endpoint_name)
        action_1_offset = self._tensor(getattr(self._cfg.insert.offset, action_1_str))
        action_2_offset = self._tensor(getattr(self._cfg.insert.offset, action_2_str))
        
        # input
        obs_all = dict(depth=[], mask=[], action_1_space=[], action_2_space=[])

        # render and store result
        for batch_idx in range(1):
            depth, mask, action_1_space, action_2_space = insert_policy_module.prepare_obs(
                depth_raw,
                mask_garment,
                mask_neckline,
                mask_hanger,
                action_1_str,
                action_2_str,
            )
            obs_all["depth"].append(depth)
            obs_all["mask"].append(mask)
            obs_all["action_1_space"].append(action_1_space)
            obs_all["action_2_space"].append(action_2_space)
        
        ret_action, inference_ret_info = inference_insert(
            module,
            obs_all,
            self._device,
            self._dtype,
            info,
        )
        ret_info = dict(
            # action_2_z_real=self._sim_z_to_real_z(inference_ret_info["action_2_z"], real_reproject_camera_z)
        )
        ret_action["action_1_xy"] += action_1_offset
        ret_action["action_2_xy"] += action_2_offset
        
        return dict(action=ret_action, info=ret_info)
    
    def _keypoints_add_offset(self, xyl: torch.Tensor, xyr: torch.Tensor, offset: omegaconf.DictConfig):
        d = xyr - xyl
        d /= torch.clamp_min(d.norm(dim=1, keepdim=True), 1e-7)
        dx, dy = d[:, 0], d[:, 1]
        nx, ny = +dy, -dx
        n = torch.concat([nx[:, None], ny[:, None]], dim=1)
        xyl += n * offset.n - d * offset.d
        xyr += n * offset.n + d * offset.d
        return xyl, xyr
    
    def predict_keypoints_score(
        self,
        depth_raw: np.ndarray, 
        mask_raw: np.ndarray, 
    ):
        self._check_depth_raw_shape(depth_raw)
        
        depth = self._tensor(depth_raw[None, ...])
        mask = self._tensor(mask_raw[None, ...])
        i = self._kp_module.preprocess_input(depth, mask)
        global_feature, xs = self._kp_module.net.encode(i)
        pred_cls = self._kp_module.run_decode_cls(global_feature)
        return float(pred_cls)

    def predict_keypoints(
        self,
        depth_raw: np.ndarray, 
        mask_raw: np.ndarray, 
        real_reproject_camera_z: float,
        info: dict,
    ):
        self._check_depth_raw_shape(depth_raw)
        
        depth = self._tensor(depth_raw[None, ...])
        mask = self._tensor(mask_raw[None, ...])
        i = self._kp_module.preprocess_input(depth, mask)
        global_feature, xs = self._kp_module.net.encode(i)
        pred_dense, pred_ij, pred_maxval = self._kp_module.run_decode_seg(global_feature, xs)
        pred_cls = self._kp_module.run_decode_cls(global_feature)
        
        left_ij = pred_ij[:, 0, :]
        right_ij = pred_ij[:, 1, :]
        if "log_path" in info.keys():
            for batch_idx in range(1):
                output_path = os.path.join(info["log_path"], "0.pdf")
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                self._kp_module.log_img_eval(depth, mask, pred_dense, pred_ij).savefig(output_path)
                plt.close()
                np.save(
                    os.path.join(info["log_path"], "0.npy"), 
                    utils.torch_dict_to_numpy_dict(dict(depth=depth, mask=mask, pred_dense=pred_dense, pred_ij=pred_ij)), 
                    allow_pickle=True
                )
        
        left_xy = policy_utils.action_batch_ij_to_xy(left_ij, depth) # [B, 2]
        right_xy = policy_utils.action_batch_ij_to_xy(right_ij, depth) # [B, 2]
        logger.info(f"\npredict keypoints left_xy:{left_xy} right_xy:{right_xy}\n")
        left_xy, right_xy = self._keypoints_add_offset(left_xy, right_xy, self._cfg.keypoint.offset)
        logger.info(f"\npredict keypoints left_xy:{left_xy} right_xy:{right_xy}\n")
        left_z = policy_utils.action_batch_ij_to_z(left_ij, depth) # [B, ]
        right_z = policy_utils.action_batch_ij_to_z(right_ij, depth) # [B, ]

        ret_action = dict(left=left_xy, right=right_xy)
        ret_info = dict(
            left_z_real=self._sim_z_to_real_z(left_z, real_reproject_camera_z),
            right_z_real=self._sim_z_to_real_z(right_z, real_reproject_camera_z),
            pred_cls=float(pred_cls),
        )

        return dict(action=ret_action, info=ret_info)

    def get_action_e2e(
        self, 
        depth_raw: np.ndarray, 
        mask_garment: np.ndarray, 
        mask_hanger: np.ndarray, 
        robot_state: np.ndarray,
    ) -> np.ndarray:
        """
        Args:
        - depth_raw: [H, W]
        - mask_garment: [H, W]
        - mask_hanger: [H, W]
        - robot_state: [D]

        Return:
        - action: [D]
        """
        assert self._policy_type in ["act"]
        H, W = depth_raw.shape
        assert mask_garment.shape == (H, W), mask_garment.shape
        assert mask_hanger.shape == (H, W), mask_hanger.shape
        assert robot_state.shape == (20, ), robot_state.shape 

        def curr_step():
            return len(self._obs_cache) - 1
        
        def _prepare_net_input():
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
            obs_input = self._tensor(obs_input).contiguous() # [B, L, C, H, W]

            sta_input = np.array(sta_input) # [L, B, R]
            sta_input = sta_input.transpose(1, 0, 2) # [B, L, R]
            sta_input = self._tensor(sta_input).contiguous() # [B, L, R]

            return obs_input, sta_input
        
        def _inference_net(obs_input, sta_input):
            logger.info(f"inference_net robot_state:\n{utils.torch_to_numpy(sta_input[0, -1, :])}") # TODO
            with torch.no_grad():
                action_pred = self._module.forward_net(obs_input, sta_input)
            logger.info(f"inference_net action:\n{utils.torch_to_numpy(action_pred[0, 0, :])}") # TODO
            return action_pred
        
        def _get_action_to_execute():
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
            ], axis=1, dtype=np.float32)[0, :] # [20, ]
            
            # magic code
            if curr_step() < 8:
                action[18: 20] = np.array([0.05, 0.025])
            return action
        
        depth_all, mask_garment_all, mask_hanger_all = depth_raw[None, ...], mask_garment[None, ...], mask_hanger[None, ...]

        self._obs_cache.append(
            np.array([
                np.clip(depth_all, None, MAX_DEPTH) / MAX_DEPTH, 
                mask_garment_all, mask_hanger_all
            ])
        )
        # magic code
        if curr_step() < 6:
            self._obs_cache[-1][2, :, :, :] = 0

        state = robot_state[None, ...]
        self._sta_cache.append(state)

        obs_input, sta_input = _prepare_net_input()
        action_pred = _inference_net(obs_input, sta_input) # [B, P, A]
        self._act_cache.append(utils.torch_to_numpy(action_pred))

        action_to_exe = _get_action_to_execute()
        return action_to_exe
    
    def reset_e2e(self):
        self._obs_cache = []
        self._sta_cache = []
        self._act_cache = []


@ti.kernel
def _e2e_fix_depth_kernel(
    depth: ti.types.ndarray(),
    extrinsics: ti.math.mat4,
    intrinsics: ti.math.mat3,
    max_y: float,
    min_y: float,
    max_x_abs: float,
    table_z: float,
    max_depth: float,
):
    H, W = depth.shape
    model_matrix = ti.Matrix(policy_utils.model_matrix_np, dt=float)
    for i, j in ti.ndrange(H, W):
        k = i * W + j
        uvw = ti.Vector([j + 0.5, i + 0.5, 1.], dt=float)
        x, y, z = (ti.math.inverse(intrinsics) @ uvw) * depth[i, j]
        xyzw = extrinsics @ model_matrix @ ti.Vector([x, y, z, 1.], dt=float)
        if xyzw[1] > max_y or xyzw[1] < min_y:
            z1 = xyzw[2] - table_z
            z2 = extrinsics[2, 3] - table_z
            depth[i, j] *= z2 / (z2 - z1)
        
        x, y, z = (ti.math.inverse(intrinsics) @ uvw) * depth[i, j]
        xyzw = extrinsics @ model_matrix @ ti.Vector([x, y, z, 1.], dt=float)
        if ti.abs(xyzw[0]) > max_x_abs:
            depth[i, j] = max_depth


def e2e_fix_depth(
    depth: np.ndarray, 
    extrinsics: np.ndarray, 
    intrinsics: np.ndarray,
    max_y: float, 
    min_y: float, 
    max_x_abs: float, 
    table_z: float,
    max_depth: float,
):
    _e2e_fix_depth_kernel(depth, extrinsics, intrinsics, max_y, min_y, max_x_abs, table_z, max_depth)
    return depth