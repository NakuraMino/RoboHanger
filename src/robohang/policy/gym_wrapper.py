import taichi as ti

import logging
logger = logging.getLogger(__name__)

from typing import Optional, List, Callable, Literal, Tuple, Dict
import copy
import math
import json
import os
import pprint

import numpy as np
import torch
import torch.nn.functional as F

import trimesh
import trimesh.transformations as tra

import omegaconf

import robohang.common.utils as utils
import robohang.sim.so3 as so3
import robohang.sim.sim_utils as sim_utils
from robohang.agent.base_agent import BaseAgent
from robohang.env.sim_env import SimEnv
import robohang.policy.policy_utils as policy_utils


@ti.data_oriented
class BaseGym:
    def __init__(self, sim_env: SimEnv, agent: BaseAgent, gym_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(gym_cfg, omegaconf.DictConfig)
        assert isinstance(sim_env, SimEnv)
        assert isinstance(agent, BaseAgent)

        self._cfg = copy.deepcopy(gym_cfg)

        # objects
        self._sim_env = sim_env
        self._agent = agent
        self._garment = self._sim_env.garment
        
        # constants
        self.batch_size = self._sim_env.batch_size
        self.device = self._sim_env.device
        self.dtype = self._sim_env.dtype
        self.dtype_int = self._sim_env.dtype_int

        # helper
        self._B_idx = torch.arange(self.batch_size, dtype=torch.long, device=self.device)

        # state
        self._init_state = self._sim_env.get_state()
    
    @property
    def sim_env(self):
        return self._sim_env
    
    @property
    def agent(self):
        return self._agent

    @property
    def garment(self):
        return self._garment
    
    def _release_grippers(self):
        # reset gripper
        for s in ["left", "right"]:
            self._sim_env.grippers[s].set_mode("Release")
            self._sim_env.grippers[s].callback(self._sim_env.env, self._sim_env.sim, -1)
    
    def _set_gripper_target_wrap(
        self, 
        xyz: torch.Tensor, 
        primitive_name: str,
        step_name: str,
        hand_name: str,
        use_ik_init_cfg: bool, 
        xyz_c: torch.Tensor,
        **kwargs, 
    ):
        """
        use_ik_init_cfg: use `predifined ik_init_cfg` or `current cfg`
        """
        def get_init_cfg():
            try:
                return omegaconf.OmegaConf.to_container(
                    getattr(getattr(getattr(self._agent.ik_init, primitive_name), step_name), hand_name)
                )
            except omegaconf.errors.ConfigAttributeError:
                return omegaconf.OmegaConf.to_container(getattr(self._agent.ik_init.default, hand_name))
            except Exception as e:
                raise e
        logger.info(f"set_gripper_target_wrap {primitive_name} {step_name} {hand_name}:\n{xyz}\n{kwargs}")
        return self._agent.set_gripper_target(
            primitive_name, step_name, hand_name, xyz, 
            rot=self._agent.get_rpy(primitive_name, step_name, hand_name, xyz, **kwargs), 
            init_cfg=None if not use_ik_init_cfg else get_init_cfg(),
            xyz_c=xyz_c,
            **kwargs,
        )

    def _calculate_layer_offset(self, xyz: torch.Tensor, h_max: float) -> torch.Tensor:
        """
        Input: [B, 3], Output: [B, 3]

        Output: z_offset = z_garment - z
        """

        layer_info = policy_utils.calculate_garment_layer(
            self._garment.get_pos(),
            self._garment.get_f2v(),
            xyz[:, None, :2],
            self._garment.dx_eps, 
        )
        z_garment = layer_info["z_upper_max"][:, 0].clamp(
            self._sim_env.get_table_height(),
            self._sim_env.get_table_height() + h_max
        ) # [B]
        layer_num = layer_info["layer_num"][:, 0] # [B]
        z_garment[torch.where(layer_num == 0)[0]] = self._sim_env.get_table_height()[torch.where(layer_num == 0)[0]] + h_max

        offset = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        offset[:, 2] = z_garment - xyz[:, 2]
        logger.info(f"calculate_layer_offset\nz_garment:{z_garment}\nlayer_num:{layer_num}\noffset:{offset}")

        return offset
    
    def _primitive_reset_grippers(
        self,
        reset_cfg: omegaconf.DictConfig,
        primitive_name: str,
        hands_str: List[str],
        callbacks,
        xyz_l: torch.Tensor=None,
        xyz_r: torch.Tensor=None,
    ):
        xyz_c = self._calculate_xyz_c(xyz_l, xyz_r)
        if "left" in hands_str:
            info = self._set_gripper_target_wrap(
                xyz_l + torch.tensor(
                    [0., 0., reset_cfg.h_upper], dtype=self.dtype, device=self.device
                ), primitive_name, "reset", "left", use_ik_init_cfg=False, xyz_c=xyz_c, 
            )
            logger.info(f"{primitive_name} reset_left:\n{info}")
        if "right" in hands_str:
            info = self._set_gripper_target_wrap(
                xyz_r + torch.tensor(
                    [0., 0., reset_cfg.h_upper], dtype=self.dtype, device=self.device
                ), primitive_name, "reset", "right", use_ik_init_cfg=False, xyz_c=xyz_c, 
            )
            logger.info(f"{primitive_name} reset_right:\n{info}")
        
        self._sim_env.set_substep("accurate")
        self._sim_env.set_actor_speed("interp", steps=reset_cfg.steps[0])
        self._sim_env.simulate(reset_cfg.steps[0], callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        self._agent.set_robot_target_to_init_qpos()
        self._sim_env.set_actor_speed("interp", steps=reset_cfg.steps[1])
        self._sim_env.simulate(reset_cfg.steps[1], callbacks=callbacks)

        logger.debug(f"{primitive_name} reset sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")
    
    def _calculate_xyz_c(self, xyz_l: Optional[torch.Tensor], xyz_r: Optional[torch.Tensor]):
        if (xyz_l is not None) and (xyz_r is not None):
            return (xyz_l + xyz_r) / 2
        elif (xyz_l is not None):
            return xyz_l.clone()
        elif (xyz_r is not None):
            return xyz_r.clone()
        else:
            raise ValueError("xyz_l, xyz_r are both None")
    
    def _primitive_move_to_pick_points(
        self,
        pick_points_cfg: omegaconf.DictConfig,
        primitive_name: str,
        hands_str: List[str],
        callbacks,
        xyz_l: torch.Tensor=None,
        xyz_r: torch.Tensor=None,
    ):
        xyz_c = self._calculate_xyz_c(xyz_l, xyz_r)
        xyz_t = {}
        xyz_lr = dict(left=xyz_l, right=xyz_r)
        for hand_name in hands_str:
            assert hand_name in ["left", "right"]

        # move to h_upper
        for hand_name in hands_str:
            xyz_t[hand_name] = xyz_lr[hand_name] + torch.tensor([0., 0., pick_points_cfg.h_upper], dtype=self.dtype, device=self.device)
            info = self._set_gripper_target_wrap(xyz_t[hand_name], primitive_name, "pick_points", hand_name, use_ik_init_cfg=True, xyz_c=xyz_c)
            logger.info(f"{primitive_name} h_upper pick_points_{hand_name}:\n{xyz_t[hand_name]}\n{info}")
        self._sim_env.set_actor_speed("interp", steps=pick_points_cfg.steps[0])
        self._sim_env.simulate(pick_points_cfg.steps[0], callbacks=callbacks)

        # move to h_inter
        for hand_name in hands_str:
            xyz_t[hand_name] = xyz_lr[hand_name] + torch.tensor([0., 0., pick_points_cfg.h_inter], dtype=self.dtype, device=self.device)
            info = self._set_gripper_target_wrap(xyz_t[hand_name], primitive_name, "pick_points", hand_name, use_ik_init_cfg=True, xyz_c=xyz_c)
            logger.info(f"{primitive_name} h_inter pick_points_{hand_name}:\n{xyz_t[hand_name]}\n{info}")
        self._sim_env.set_actor_speed("interp", steps=pick_points_cfg.steps[1])
        self._sim_env.simulate(pick_points_cfg.steps[1], callbacks=callbacks)

        # move to (z_garment + h_delta)
        for hand_name in hands_str:
            xyz_t[hand_name] = xyz_lr[hand_name]+ torch.tensor(
                [0., 0., pick_points_cfg.h_delta], dtype=self.dtype, device=self.device
            ) + self._calculate_layer_offset(xyz_lr[hand_name], pick_points_cfg.h_inter - pick_points_cfg.h_delta)
            info = self._set_gripper_target_wrap(xyz_t[hand_name], primitive_name, "pick_points", hand_name, use_ik_init_cfg=True, xyz_c=xyz_c)
            logger.info(f"{primitive_name} h_delta pick_points_{hand_name}:\n{xyz_t[hand_name]}\n{info}")
        self._sim_env.set_actor_speed("interp", steps=pick_points_cfg.steps[2])
        self._sim_env.simulate(pick_points_cfg.steps[2], callbacks=callbacks)

        # pick
        for hand_name in hands_str:
            self._sim_env.grippers[hand_name].set_mode("Pick")
            self._sim_env.grippers[hand_name].callback(self._sim_env.env, self._sim_env.sim, -1)
            self._sim_env.grippers[hand_name].set_mode("Hold")

        # move away
        for hand_name in hands_str:
            xyz_t[hand_name] = torch.max(
                xyz_lr[hand_name] + torch.tensor([0., 0., pick_points_cfg.h_later], dtype=self.dtype, device=self.device),
                xyz_t[hand_name] + torch.tensor([0., 0., getattr(pick_points_cfg, "h_later_min", 0.)], dtype=self.dtype, device=self.device),
            )
            info = self._set_gripper_target_wrap(xyz_t[hand_name], primitive_name, "pick_points", hand_name, use_ik_init_cfg=True, xyz_c=xyz_c)
            logger.info(f"{primitive_name} h_later pick_points_{hand_name}:\n{xyz_t[hand_name]}\n{info}")
        self._sim_env.set_actor_speed("interp", steps=pick_points_cfg.steps[3])
        self._sim_env.simulate(pick_points_cfg.steps[3], callbacks=callbacks)

        logger.debug(f"{primitive_name} pick sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")
    
    def domain_randomize(self):
        self._sim_env.domain_randomize()
        
    def get_state(self):
        return dict(
            sim_env=self._sim_env.get_state(),
        )
    
    def set_state(self, state: dict):
        self._sim_env.set_state(state["sim_env"])

    def reset(self):
        self._sim_env.set_state(self._init_state)