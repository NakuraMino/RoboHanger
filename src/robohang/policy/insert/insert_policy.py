import logging
logger = logging.getLogger(__name__)

import copy
from typing import Dict, Literal, Any, List, Callable, Tuple
import time
import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt

import omegaconf

import robohang.common.utils as utils
import robohang.sim.sim_utils as sim_utils
import robohang.sim.so3 as so3
import robohang.policy.policy_utils as policy_utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
from robohang.policy.insert.insert_gym import InsertGym
from robohang.policy.insert.insert_learn import (
    InsertLeftEndModule, InsertRightEndModule, DualArmModule, concat_mask
)


class InsertPolicy:
    def __init__(self, insert_gym: InsertGym, policy_cfg: omegaconf.DictConfig) -> None:
        self._policy_cfg = copy.deepcopy(policy_cfg)
        self._insert_gym = insert_gym
        self._agent = insert_gym.agent

        self.dtype = self._insert_gym.dtype
        self.dtype_int = self._insert_gym.dtype_int
        self.device = self._insert_gym.device
        self.batch_size = self._insert_gym.batch_size

        self._B_idx = torch.arange(self.batch_size, dtype=self.dtype_int, device=self.device)

    def get_lift_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._insert_gym.random_lift_action(),
            info=dict(),
        )
    
    def get_press_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._insert_gym.random_press_action(),
            info=dict(),
        )

    def get_drag_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._insert_gym.random_drag_action(),
            info=dict(),
        )
    
    def get_rotate_action(self) -> Dict[Literal["action", "info"], Any]:
        return dict(
            action=self._insert_gym.random_rotate_action(),
            info=dict(),
        )


class InsertPolicyRandom(InsertPolicy):
    def __init__(self, insert_gym: InsertGym, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)

    def _get_action(self, action_cfg: omegaconf.DictConfig, random_action_func: Callable) -> Dict[Literal["action", "info"], Any]:
        per_batch = action_cfg.query.per_batch
        num_iter = action_cfg.query.num_iter
        dx_eps = self._insert_gym.garment.dx_eps

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
                x["xy"],
            ], dim=1).view(self.batch_size, per_batch, 2)
        
        def unpack_sample(x: torch.Tensor):
            return dict(
                xy=x[..., :2],
            )
        
        def generate_sample_func():
            return pack_sample(random_action_func(self.batch_size * per_batch))
        
        def is_good_sample_func(sample: torch.Tensor):
            """sample: [B, D, 2]"""
            unpacked = unpack_sample(sample)
            xy = unpacked["xy"]

            pos = self._insert_gym.garment.get_pos()
            f2v = self._insert_gym.garment.get_f2v()

            satisfy_rule = torch.ones((self.batch_size, per_batch), dtype=self.dtype_int, device=self.device)
            for rule in action_cfg.heuristic:
                if len(rule) == 4:
                    x, y, f, n = rule
                    assert f in ["eq", "ge", "le"], f
                    qxy = xy + torch.tensor([[x, y]], dtype=self.dtype, device=self.device)
                    layer_num = policy_utils.calculate_garment_layer(pos, f2v, qxy, dx_eps)["layer_num"] # [B, D]
                    satisfy_rule = torch.logical_and(satisfy_rule, getattr(torch, f)(layer_num, n))
                elif len(rule) == 3 and rule[-1] == "hanger":
                    hmin, hmax, name = rule
                    hanger_matinv = so3.pos7d_to_matinv(self._insert_gym.sim_env.hanger.get_pos())[:, None, :, :] # [B, 1, 4, 4]
                    xyzw = torch.zeros((self.batch_size, per_batch, 4), dtype=self.dtype, device=self.device) # [B, D, 4]
                    xyzw[:, :, 3] = 1.
                    xyzw[:, :, :2] = xy
                    h = (hanger_matinv @ xyzw[..., None])[:, :, 1, 0]
                    satisfy_rule = torch.logical_and(
                        torch.logical_and(
                            hmin <= h, h <= hmax,
                        ), satisfy_rule, 
                    )
                elif len(rule) == 5 and rule[-1] == "press":
                    xmin, xmax, ymin, ymax, name = rule
                    press_xy = self._insert_gym.get_current_xyz("left")[:, None, :2] # [B, 1, 2]
                    dxy = xy - press_xy # [B, D, 2]
                    satisfy_rule = torch.logical_and(
                        torch.logical_and(
                            torch.logical_and(xmin <= dxy[:, :, 0], dxy[:, :, 0] <= xmax),
                            torch.logical_and(ymin <= dxy[:, :, 1], dxy[:, :, 1] <= ymax),
                        ), satisfy_rule, 
                    )

                else:
                    raise ValueError(rule)

            is_good_sample = torch.logical_or(
                torch.logical_and(target_B1 == idx_dict["fail"], torch.logical_not(satisfy_rule)), # fail
                torch.logical_and(target_B1 == idx_dict["success"], satisfy_rule),
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
        return dict(action=unpack_sample(current_answer), info=info)

    def get_lift_action(self):
        action_info = self._get_action(
            self._policy_cfg.lift,
            self._insert_gym.random_lift_action,
        )
        logger.info(f"get_lift_action info:\n{action_info['info']}")
        return action_info
    
    def get_press_action(self):
        action_info = self._get_action(
            self._policy_cfg.press,
            self._insert_gym.random_press_action,
        )
        logger.info(f"get_press_action info:\n{action_info['info']}")
        return action_info
    
    def get_drag_action(self):
        action_info = self._get_action(
            self._policy_cfg.drag,
            self._insert_gym.random_drag_action,
        )
        logger.info(f"get_drag_action info:\n{action_info['info']}")
        return action_info


def prepare_obs(
    depth_raw: np.ndarray, 
    mask_garment: np.ndarray,
    mask_inverse: np.ndarray,
    mask_hanger: np.ndarray,
    action_1_str: str,
    action_2_str: str,
):
    mask_raw = concat_mask(dict(mask_garment=mask_garment, mask_inverse=mask_inverse, mask_hanger=mask_hanger))

    action_1_space = np.zeros_like(depth_raw)
    i, j = MetaInfo.get_action_space_slice(action_1_str, depth_raw)
    action_1_space[i, j] = 1.
    action_2_space = np.zeros_like(depth_raw)
    i, j = MetaInfo.get_action_space_slice(action_2_str, depth_raw)
    action_2_space[i, j] = 1.

    return depth_raw, mask_raw, action_1_space, action_2_space


def inference_insert(
    module: DualArmModule,
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

    action_1_ij = dense_info["action_1_ij"]
    action_1_z = policy_utils.action_batch_ij_to_z(action_1_ij, depth)
    ret_info["action_1_z"] = action_1_z

    return ret_action, ret_info


class InsertPolicyUNet(InsertPolicy):
    def __init__(self, insert_gym: InsertGym, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)

        self._le_module = InsertLeftEndModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt.left), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()

        self._re_module = InsertRightEndModule.load_from_checkpoint(
            utils.get_path_handler()(self._policy_cfg.ckpt.right), 
            map_location=torch.device("cpu"),
        ).to(self.device).eval()

        self._cache = dict()
    
    def _net_inference(
        self, 
        endpoint_name: Literal["left", "right"],
        info: dict,
    ):
        module = dict(left=self._le_module, right=self._re_module)[endpoint_name]
        action_1_str: str = module.action_1_str
        action_2_str: str = module.action_2_str
        
        # input
        obs_all = dict(depth=[], mask=[], action_1_space=[], action_2_space=[])

        # render and store result
        for batch_idx in range(self.batch_size):
            result, mask_str_to_idx, camera_info = self._agent.get_obs("direct", "small", batch_idx, True, pos=self._agent.direct_obs)
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=self._policy_cfg.interp_mask, target="double_side")
            depth_raw = reproject_result["depth_output"]
            mask_garment = reproject_result["mask_output"]
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=self._policy_cfg.interp_mask, target="inverse_side")
            mask_inverse = reproject_result["mask_output"]
            reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, interp_mask=self._policy_cfg.interp_mask, target="hanger")
            mask_hanger = reproject_result["mask_output"]

            depth_raw, mask_raw, action_1_space, action_2_space = prepare_obs(
                depth_raw,
                mask_garment,
                mask_inverse,
                mask_hanger,
                action_1_str,
                action_2_str,
            )
            obs_all["depth"].append(depth_raw)
            obs_all["mask"].append(mask_raw)
            obs_all["action_1_space"].append(action_1_space)
            obs_all["action_2_space"].append(action_2_space)
        
        ret_action, ret_info = inference_insert(
            module,
            obs_all,
            self.device,
            self.dtype,
            info,
        )

        return dict(action=ret_action, info=ret_info)
    
    def get_press_action(self, info: dict):
        action_and_info = self._net_inference("left", info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_1_xy"])
        self._cache["press"] = action_and_info["action"]["action_2_xy"]
        return dict(action=action, info=action_and_info["info"])
    
    def get_lift_action(self, info: dict):
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=self._cache.pop("press"))
        return dict(action=action, info=dict())
    
    def get_drag_action(self, info: dict):
        action_and_info = self._net_inference("right", info)
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=action_and_info["action"]["action_1_xy"])
        self._cache["rotate"] = action_and_info["action"]["action_2_xy"]
        return dict(action=action, info=action_and_info["info"])
    
    def get_rotate_action(self, info: dict):
        action: Dict[Literal["xy"], torch.Tensor] = dict(xy=self._cache.pop("rotate"))
        return dict(action=action, info=dict())