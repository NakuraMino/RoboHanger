import taichi as ti

import logging
logger = logging.getLogger(__name__)

from typing import Optional, List, Callable, Literal, Tuple, Dict
import copy
import math
import json
import os
import pprint
from dataclasses import asdict

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
from robohang.policy.gym_wrapper import BaseGym


@ti.func
def _xy2ji_func(xy: ti.math.vec2, xy_range: ti.math.mat2, wh: ti.math.ivec2) -> ti.math.ivec2:
    return ti.math.clamp(
        ti.floor((xy - xy_range[0, :]) / (xy_range[1, :] - xy_range[0, :]) * wh, dtype=int),
        xmin=0, xmax=wh,
    )


@ti.func
def _ji2xy_func(ji: ti.math.vec2, xy_range: ti.math.mat2, wh: ti.math.ivec2) -> ti.math.vec2:
    return xy_range[0, :] + (ji + 0.5) * (xy_range[1, :] - xy_range[0, :]) / wh


@ti.kernel
def _calculate_atractable_keypoints_kernel(
    vert: ti.types.ndarray(dtype=ti.math.vec3),
    vert_rest: ti.types.ndarray(dtype=ti.math.vec3),
    vid_lr: ti.math.ivec2,
    xy_range: ti.types.ndarray(ti.math.mat2),
    ans: ti.types.ndarray(dtype=ti.math.ivec2),
    grasp_wrong: ti.types.ndarray(dtype=ti.math.ivec2),
    gripper_radius: float,
    dist_threshold: float,
):
    """
    Args:
    - vert: [B, V][3]
    - vert_rest: [V][3]
    - vid_lr: ivec2 [vid_left, vid_right]
    - xy_range: mat2, [[xmin, ymin], [xmax, ymax]]
    - ans: [B, H, W][2]
    - grasp_wrong: [B, H, W][2]
    """
    B, H, W = ans.shape
    V, = vert_rest.shape
    WH = ti.Vector([W, H], dt=int)

    for b, h, w in ti.ndrange(B, H, W):
        ans[b, h, w] = 0
        grasp_wrong[b, h, w] = 0
    
    loop_size = B * V * 2
    if loop_size > 2e9:
        print(f"[WARN] loop size {loop_size} is too large in calculate_atractable_keypoints_kernel")
    
    for b, v, lr_idx in ti.ndrange((0, B), (0, V), (0, 2)):
        radius_j, radius_i = ti.ceil(WH * dist_threshold / (xy_range[b][1, :] - xy_range[b][0, :]), dtype=int)
        for dj, di in ti.ndrange((-radius_j, radius_j + 1), (-radius_i, radius_i + 1)):
            vid = vid_lr[lr_idx]
            vid_xy = vert[b, vid][:2]
            vid_ji = _xy2ji_func(vid_xy, xy_range[b], WH)
            ji = vid_ji + ti.Vector([dj, di], dt=int)
            xy = _ji2xy_func(ji, xy_range[b], WH)
            
            if (0 <= ji).all() and (ji < WH).all() and ti.math.length(xy - vid_xy) < dist_threshold: # this 'if' is only a quick filter
                # since xy is near keypoint[lr_idx], (x, y) has potential to success
                # check whether (x, y) satisfy our rules
                if ti.math.length(vert[b, v][:2] - xy) < gripper_radius:
                    if ti.math.length(vert_rest[v] - vert_rest[vid]) > dist_threshold:
                        grasp_wrong[b, H - 1 - ji[1], ji[0]][lr_idx] = 1
                    else:
                        ans[b, H - 1 - ji[1], ji[0]][lr_idx] = 1
    
    for b, h, w, lr_idx in ti.ndrange(B, H, W, 2):
        if grasp_wrong[b, H - 1 - h, w][lr_idx] == 1:
            ans[b, H - 1 - h, w][lr_idx] = 0


def clip_fling_action(
    action_space_cfg: omegaconf.DictConfig,
    center_xy: torch.Tensor,
    distance: torch.Tensor,
    angle_degree: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
    eps=1e-6,
):
    action_space_center_xy = torch.tensor(np.array(action_space_cfg.center), dtype=dtype, device=device)
    center_delta = center_xy - action_space_center_xy # [B, 2]
    center_delta_norm = torch.norm(center_delta, dim=1, keepdim=True) # [B, 2]
    center_xy = action_space_center_xy + center_delta / torch.clamp_min(center_delta_norm, eps) * torch.clamp_max(center_delta_norm, float(action_space_cfg.radius))
    distance = torch.clip(
        distance, 
        min=action_space_cfg.distance[0],
        max=action_space_cfg.distance[1],
    )
    angle_degree = torch.clip(
        angle_degree, 
        min=action_space_cfg.angle_degree[0],
        max=action_space_cfg.angle_degree[1],
    )
    return center_xy, distance, angle_degree


@ti.data_oriented
class FunnelGym(BaseGym):
    def __init__(self, sim_env: SimEnv, agent: BaseAgent, gym_cfg: omegaconf.DictConfig) -> None:
        super().__init__(sim_env, agent, gym_cfg)

        # primitive hyperparameters
        self._parameter: omegaconf.DictConfig = copy.deepcopy(self._cfg.parameter)
        self._init_garment_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.init_garment)
        self._fling_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.fling)
        self._pick_place_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.pick_place)

        # reward (coverage)
        self._reward_coverage_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.reward.coverage)
        self._garment_rest_area = float(self._garment.rest_mesh.area / 2.)
        self._coverage_grid = torch.zeros(
            (self.batch_size, self._reward_coverage_cfg.ny, self._reward_coverage_cfg.nx), 
            dtype=self.dtype_int, device=self.device,
        )
        """[B, NY, NX]"""

        # reward (orientation)
        self._reward_orientation_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.reward.orientation)
        self._keypoints_vids = [self._sim_env.garment_keypoints[s] for s in self._reward_orientation_cfg.keypoints_name]
        mesh = trimesh.Trimesh(vertices=self._garment.rest_mesh.vertices, faces=self._garment.rest_mesh.faces)
        mesh.apply_transform(
            tra.translation_matrix(self._reward_orientation_cfg.target.xyz) @
            tra.euler_matrix(*(self._reward_orientation_cfg.target.rpy))
        )
        self._keypoints_target = torch.tensor(
            mesh.vertices[self._keypoints_vids, :],
            dtype=self.dtype, device=self.device,
        )
        """[K, 3]"""
        logger.info(f"orientation_target:\n{self._keypoints_target}")

        # state
        self._init_state = self._sim_env.get_state()
    
    @property
    def fling_normal_action_space(self) -> omegaconf.DictConfig:
        return self._fling_cfg.normal.action_space

    def _damp_callback(self, *args, **kwargs):
        vel = self._garment.get_vel()
        vel *= (1. - self._parameter.damp_per_substep)
        self._garment.set_vel(vel)

    def primitive_init_garment(self, callbacks: Optional[List[Callable]]=None):
        '''print("[WARN] debug", __name__)
        return'''
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")
        
        nv = self._garment.nv
        xyz = torch.tensor([
            self._init_garment_cfg.x, 
            self._init_garment_cfg.y, 
            self._init_garment_cfg.z
        ], dtype=self.dtype, device=self.device)

        up_xy = torch.tensor([
            self._init_garment_cfg.up_x,
            self._init_garment_cfg.up_y,
        ], dtype=self.dtype, device=self.device)

        random_vid = torch.randint(0, nv, (self.batch_size, ), dtype=torch.long, device=self.device) # [B, ], long
        random_xyz = torch.rand((self.batch_size, 3), dtype=self.dtype, device=self.device)
        random_xyz = utils.map_01_ab(random_xyz, xyz[:, 0], xyz[:, 1]) # [B, 3], float
        random_rot = torch.tensor(np.array([tra.random_rotation_matrix(num=1) for _ in range(self.batch_size)]), dtype=self.dtype, device=self.device) # [B, 4, 4], float
        logger.info(f"\nrandom_vid:{random_vid}\nrandom_xyz:{random_xyz}\nrandom_rot:{random_rot}")

        # reset garment
        self._garment.reset()

        # set pos
        pos = self._garment.get_pos() # [B, V, 3], float
        rotated = (random_rot[:, None, :3, :3] @ pos[:, :, :, None])[:, :, :, 0]
        translation = torch.zeros_like(random_xyz) # [B, 3], float
        translation[:, 2] = random_xyz[:, 2] - rotated[self._B_idx, random_vid, 2]
        translation[:, :2] = up_xy - rotated[:, :, :2].mean(dim=1)
        rotated_translated = rotated + translation[:, None, :]
        self._garment.set_pos(rotated_translated)

        # get constraints
        con = self._garment.get_constraint()
        
        # hold, wait to fall
        con[self._B_idx, random_vid, :] = 0.0
        self._garment.set_constraint(con)
        self._sim_env.set_substep("accurate")
        self._sim_env.simulate(self._init_garment_cfg.steps[0], callbacks=[self._damp_callback] + callbacks)
        self._sim_env.set_substep("efficient")

        self._sim_env.simulate(self._init_garment_cfg.steps[1], callbacks=callbacks)

        # release, wait to fall
        con[self._B_idx, random_vid, :] = 1.0
        self._garment.set_constraint(con)

        pos = self._garment.get_pos() # [B, V, 3], float
        translation = torch.zeros_like(random_xyz) # [B, 3], float
        translation[:, :2] = up_xy - pos[:, :, :2].mean(dim=1)
        self._garment.set_pos(pos + translation[:, None, :])

        self._sim_env.set_substep("accurate")
        self._sim_env.simulate(self._init_garment_cfg.steps[2], callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        self._sim_env.simulate(self._init_garment_cfg.steps[3], callbacks=callbacks)

        # translate
        pos = self._garment.get_pos()
        translation = torch.zeros_like(random_xyz) # [B, 3], float
        translation[:, :2] = random_xyz[:, :2] - pos[:, :, :2].mean(dim=1)
        self._garment.set_pos(pos + translation[:, None, :])

        self._sim_env.simulate(self._init_garment_cfg.steps[4], callbacks=callbacks)

        logger.debug(f"init_garment sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")

    def _pickpoints_liftup_flingforward_flingbackward_reset(
        self, 
        xyz_l: torch.Tensor,
        xyz_r: torch.Tensor,
        pick_points_cfg: omegaconf.DictConfig,
        lift_up_cfg: omegaconf.DictConfig,
        fling_forwards_cfg: List[omegaconf.DictConfig],
        reset_cfg: omegaconf.DictConfig,
        callbacks: List[Callable],
    ):
        primitive_name = "fling"

        # move to pick points
        self._primitive_move_to_pick_points(
            pick_points_cfg,
            primitive_name,
            ["left", "right"],
            callbacks,
            xyz_l=xyz_l,
            xyz_r=xyz_r,
        )
        
        # lift up
        self._sim_env.set_substep("accurate")
        dist = (xyz_l - xyz_r).norm(dim=-1)

        lift_l = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        lift_l[:, 0] = -dist * lift_up_cfg.x.scale / 2.
        lift_l[:, 1] = lift_up_cfg.y
        lift_l[:, 2] = lift_up_cfg.h + self._sim_env.get_table_height()
        
        lift_r = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        lift_r[:, 0] = +dist * lift_up_cfg.x.scale / 2.
        lift_r[:, 1] = lift_up_cfg.y
        lift_r[:, 2] = lift_up_cfg.h + self._sim_env.get_table_height()

        xyz_c = self._calculate_xyz_c(lift_l, lift_r)
        info = self._set_gripper_target_wrap(
            lift_l, primitive_name, "lift_up", "left", use_ik_init_cfg=True, xyz_c=xyz_c, 
        )
        logger.info(f"{primitive_name} lift_up_lower_left:\n{info}")
        info = self._set_gripper_target_wrap(
            lift_r, primitive_name, "lift_up", "right", use_ik_init_cfg=True, xyz_c=xyz_c, 
        )
        logger.info(f"{primitive_name} lift_up_lower_right:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=lift_up_cfg.steps[0])
        self._sim_env.simulate(lift_up_cfg.steps[0], callbacks=[self._damp_callback] + callbacks)
        self._sim_env.set_substep("efficient")

        logger.debug(f"{primitive_name} lift sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")

        # fling forward
        '''self._sim_env.set_substep("accurate")'''
        '''for interp_idx, interp in enumerate(fling_backward_cfg.interp):'''
        '''fling_l = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        fling_l[:, 0] = lift_l[:, 0]
        fling_l[:, 1] = fling_backward_cfg.y
        fling_l[:, 2] = self._sim_env.get_table_height() + fling_backward_cfg.h

        fling_r = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        fling_r[:, 0] = lift_r[:, 0]
        fling_r[:, 1] = fling_backward_cfg.y
        fling_r[:, 2] = self._sim_env.get_table_height() + fling_backward_cfg.h

        xyz_c = self._calculate_xyz_c(fling_l, fling_r)
        info = self._set_gripper_target_wrap(
            fling_l, primitive_name, "fling_backward", "left", use_ik_init_cfg=True, xyz_c=xyz_c, 
        )
        logger.info(f"{primitive_name} fling_backward_left:\n{info}")
        info = self._set_gripper_target_wrap(
            fling_r, primitive_name, "fling_backward", "right", use_ik_init_cfg=True, xyz_c=xyz_c, 
        )
        logger.info(f"{primitive_name} fling_backward_right:\n{info}")

        logger.info(f"fling_backward_residual:{pprint.pformat(self._sim_env.actor.get_residual(), sort_dicts=False)}")
        logger.info(f"fling_backward_curr_qpos:{pprint.pformat(self._sim_env.robot.get_cfg_pos(), sort_dicts=False)}")
        logger.info(f"fling_backward_target_qpos:{pprint.pformat(self._sim_env.actor.get_target(), sort_dicts=False)}")

        steps = fling_backward_cfg.steps
        self._sim_env.set_actor_speed("interp", steps=steps)
        self._sim_env.simulate(steps, callbacks=callbacks)
        if hasattr(fling_backward_cfg, "wait"):
            self._sim_env.simulate(fling_backward_cfg.wait, callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        logger.debug(f"{primitive_name} fling_backward sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")'''
        
        # fling forward
        self._sim_env.set_substep("accurate")
        for fling_forward_idx, fling_forward_cfg in enumerate(fling_forwards_cfg):
            fling_forward_str = f"fling_forward_{fling_forward_idx}"
            fling_l = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
            fling_l[:, 0] = lift_l[:, 0]
            fling_l[:, 1] = fling_forward_cfg.y
            fling_l[:, 2] = self._sim_env.get_table_height() + fling_forward_cfg.h

            fling_r = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
            fling_r[:, 0] = lift_r[:, 0]
            fling_r[:, 1] = fling_forward_cfg.y
            fling_r[:, 2] = self._sim_env.get_table_height() + fling_forward_cfg.h

            xyz_c = self._calculate_xyz_c(fling_l, fling_r)
            info = self._set_gripper_target_wrap(
                fling_l, primitive_name, fling_forward_str, "left", use_ik_init_cfg=True, xyz_c=xyz_c, 
            )
            logger.info(f"{primitive_name} {fling_forward_str}_left:\n{info}")
            info = self._set_gripper_target_wrap(
                fling_r, primitive_name, fling_forward_str, "right", use_ik_init_cfg=True, xyz_c=xyz_c, 
            )
            logger.info(f"{primitive_name} {fling_forward_str}_right:\n{info}")

            logger.info(f"{fling_forward_str}_residual:{pprint.pformat(self._sim_env.actor.get_residual(), sort_dicts=False)}")
            logger.info(f"{fling_forward_str}_curr_qpos:{pprint.pformat(self._sim_env.robot.get_cfg_pos(), sort_dicts=False)}")
            logger.info(f"{fling_forward_str}_target_qpos:{pprint.pformat(self._sim_env.actor.get_target(), sort_dicts=False)}")

            steps = fling_forward_cfg.steps
            self._sim_env.set_actor_speed("interp", steps=steps)
            self._sim_env.simulate(steps, callbacks=callbacks)

            # release gripper at last step
            if fling_forward_idx == len(fling_forwards_cfg) - 1:
                self._release_grippers()

            if hasattr(fling_forward_cfg, "wait"):
                self._sim_env.simulate(fling_forward_cfg.wait, callbacks=callbacks)

        logger.debug(f"{primitive_name} fling_forward sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")

        # reset gripper
        self._primitive_reset_grippers(
            reset_cfg,
            primitive_name,
            ["left", "right"],
            callbacks,
            xyz_l=fling_l,
            xyz_r=fling_r,
        )

    def primitive_fling(
        self, 
        center_xy: torch.Tensor, 
        distance: torch.Tensor, 
        angle_degree: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """center_xy: [B, 2]; distance, angle: [B, ]"""
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_fling raw:\n{center_xy}\n{distance}\n{angle_degree}\n")
        assert center_xy.shape == (self.batch_size, 2)
        assert distance.shape == (self.batch_size, )
        assert angle_degree.shape == (self.batch_size, )

        # clip action space (normal)
        center_xy, distance, angle_degree = clip_fling_action(
            self._fling_cfg.normal.action_space,
            center_xy, distance, angle_degree, self.dtype, self.device,
        )

        xyz_c = F.pad(center_xy, (0, 1), "constant", 0)
        xyz_c[:, 2] = self._sim_env.get_table_height()

        # center to left and right
        xyz_l = xyz_c.clone()
        xyz_l[:, 0] -= torch.cos(torch.deg2rad(angle_degree)) * distance / 2.
        xyz_l[:, 1] -= torch.sin(torch.deg2rad(angle_degree)) * distance / 2.

        xyz_r = xyz_c.clone()
        xyz_r[:, 0] += torch.cos(torch.deg2rad(angle_degree)) * distance / 2.
        xyz_r[:, 1] += torch.sin(torch.deg2rad(angle_degree)) * distance / 2.

        logger.info(f"primitive_fling clipped:\n{xyz_l}\n{xyz_r}\n{xyz_c}")
        
        self._pickpoints_liftup_flingforward_flingbackward_reset(
            xyz_l, 
            xyz_r, 
            self._fling_cfg.key_points.pick_points,
            self._fling_cfg.key_points.lift_up,
            self._fling_cfg.key_points.fling_forwards,
            self._fling_cfg.key_points.reset,
            callbacks,
        )

    def random_fling(self, num: Optional[int]=None) -> Dict[Literal["center_xy", "distance", "angle_degree"], torch.Tensor]:
        if num is None:
            num = self.batch_size
        num = int(num)
        
        action_space_cfg = self._fling_cfg.normal.action_space
        center_xy_angle = utils.map_01_ab(
            torch.rand((num, 1), dtype=self.dtype, device=self.device), 
            0., torch.pi * 2.,
        )
        center_xy_radius = torch.sqrt(
            torch.rand((num, 1), dtype=self.dtype, device=self.device)
        ) * float(action_space_cfg.radius)
        normal = dict(
            center_xy=torch.concat([
                center_xy_radius * torch.cos(center_xy_angle),
                center_xy_radius * torch.sin(center_xy_angle),
            ], dim=1) + torch.tensor(np.array(action_space_cfg.center), dtype=self.dtype, device=self.device),
            distance=utils.map_01_ab(
                torch.rand((num, ), dtype=self.dtype, device=self.device), 
                action_space_cfg.distance[0], action_space_cfg.distance[1],
            ),
            angle_degree=utils.map_01_ab(
                torch.rand((num, ), dtype=self.dtype, device=self.device), 
                action_space_cfg.angle_degree[0], action_space_cfg.angle_degree[1],
            ),
        )
        return normal

    def primitive_pick_place(
        self, 
        xy_s: torch.Tensor, 
        xy_e: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy_s, xy_e: [B, 2]"""
        '''"""xy_s: [B, 2], distance, angle_degree: [B, ]"""'''
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_pick_place raw:\n{xy_s}\n{xy_e}")
        assert xy_s.shape == xy_e.shape == (self.batch_size, 2)

        primitive_name = "pick_place"

        # clip action space
        xyz_s = F.pad(torch.clip(
            xy_s.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._pick_place_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._pick_place_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz_s[:, 2] = self._sim_env.get_table_height()

        xyz_e = F.pad(torch.clip(
            xy_e.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._pick_place_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._pick_place_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz_e[:, 2] = self._sim_env.get_table_height()
        
        logger.info(f"primitive_pick_place clipped:\n{xyz_s}\n{xyz_e}")

        # pick
        self._primitive_move_to_pick_points(
            self._pick_place_cfg.pick_points,
            primitive_name,
            ["left"],
            callbacks,
            xyz_l=xyz_s,
        )
        
        # drag
        self._sim_env.set_substep("accurate")
        xyz_c = self._calculate_xyz_c(xyz_e, None)
        info = self._set_gripper_target_wrap(
            xyz_e + torch.tensor(
                [0., 0., self._pick_place_cfg.pick_points.h_later], dtype=self.dtype, device=self.device,
            ), primitive_name, "pick_points", "left", use_ik_init_cfg=True, xyz_c=xyz_c,
        )
        logger.info(f"[IMPORTANT] pick_place_e:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._pick_place_cfg.drag_step[0])
        self._sim_env.simulate(self._pick_place_cfg.drag_step[0], callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        for s in ["left"]:
            self._sim_env.grippers[s].set_mode("Release")
            self._sim_env.grippers[s].callback(self._sim_env.env, self._sim_env.sim, -1)
        self._sim_env.simulate(self._pick_place_cfg.drag_step[1], callbacks=callbacks)

        logger.debug(f"{primitive_name} drag sim_error {pprint.pformat(self._sim_env.get_sim_error(), sort_dicts=False)}")

        # reset gripper
        self._primitive_reset_grippers(
            self._pick_place_cfg.reset,
            primitive_name,
            ["left"],
            callbacks,
            xyz_l=xyz_e,
        )
    
    def random_pick_place(self, num: Optional[int]=None) -> Dict[Literal["xy_s", "xy_e"], torch.Tensor]:
        if num is None:
            num = self.batch_size
        num = int(num)

        action_space_cfg = self._pick_place_cfg.action_space
        return dict(
            xy_s=torch.concat([
                utils.map_01_ab(
                    torch.rand((num, 1), dtype=self.dtype, device=self.device), 
                    action_space_cfg.min[0], action_space_cfg.max[0],
                ),
                utils.map_01_ab(
                    torch.rand((num, 1), dtype=self.dtype, device=self.device),
                    action_space_cfg.min[1], action_space_cfg.max[1],
                ),
            ], dim=1),
            xy_e=torch.concat([
                utils.map_01_ab(
                    torch.rand((num, 1), dtype=self.dtype, device=self.device), 
                    action_space_cfg.min[0], action_space_cfg.max[0],
                ),
                utils.map_01_ab(
                    torch.rand((num, 1), dtype=self.dtype, device=self.device),
                    action_space_cfg.min[1], action_space_cfg.max[1],
                ),
            ], dim=1),
        )
    
    def _keypoints_clip_input(
        self, 
        xy1: torch.Tensor, xy2: torch.Tensor, 
        dist_range: List[float], xyc: List[float], radius_max: float
    ):
        # process xym
        xym = (xy1 + xy2) / 2.
        xyc = torch.tensor(xyc, dtype=self.dtype, device=self.device)
        xym_to_xyc = xym - xyc
        xym_to_xyc_len = xym_to_xyc.norm(dim=1)
        n = xym_to_xyc / xym_to_xyc_len[:, None]
        too_far_indices = torch.where(xym_to_xyc_len > radius_max)[0]
        xym_to_xyc_new = xym_to_xyc.clone()
        xym_to_xyc_new[too_far_indices, :] = n[too_far_indices, :] * radius_max
        xym_new = xym_to_xyc_new + xyc
        
        # process xyd
        xyd = (xy2 - xy1)
        eps = 1e-5
        xyd_len = xyd.norm(dim=1)
        n = xyd / xyd_len[:, None]
        n[torch.where(xyd_len < eps)[0], :] = torch.tensor([0., 1.], dtype=self.dtype, device=self.device)
        xyd_new = n * torch.clamp(xyd_len, min(dist_range), max(dist_range))[:, None]

        return xym_new - xyd_new / 2, xym_new + xyd_new / 2

    def primitive_keypoints(
        self,
        xy_l: torch.Tensor,
        xy_r: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_keypoints raw:\n{xy_l}\n{xy_r}")
        assert xy_l.shape == xy_r.shape == (self.batch_size, 2)

        keypoint_cfg = self._cfg.primitive.keypoints

        # clip action space (keypoints)
        xy_l, xy_r = self._keypoints_clip_input(
            xy1=xy_l, xy2=xy_r,
            **(omegaconf.OmegaConf.to_container(keypoint_cfg.action_space))
        )

        # decide whether to change hand to avoid self collision
        xy_c = (xy_l + xy_r) / 2.
        l_to_r = xy_r - xy_l
        theta1 = torch.atan2(xy_c[:, 1], xy_c[:, 0])
        theta2 = torch.atan2(l_to_r[:, 1], l_to_r[:, 0])
        theta3 = (theta2  - theta1 + 4 * math.pi) % (2 * math.pi)
        exchange_hand = torch.logical_and(theta3 > math.radians(40), theta3 < math.radians(140))
        xy_l_clone, xy_r_clone = xy_l.clone(), xy_r.clone()
        xy_l = torch.where(exchange_hand[:, None], xy_r_clone, xy_l_clone)
        xy_r = torch.where(exchange_hand[:, None], xy_l_clone, xy_r_clone)
        logger.info(f"primitive_keypoints:\n{xy_l}\n{xy_r}")

        xyz_l = F.pad(xy_l, (0, 1), "constant", 0)
        xyz_l[:, 2] = self._sim_env.get_table_height()

        xyz_r = F.pad(xy_r, (0, 1), "constant", 0)
        xyz_r[:, 2] = self._sim_env.get_table_height()
        
        self._pickpoints_liftup_flingforward_flingbackward_reset(
            xyz_l, 
            xyz_r, 
            keypoint_cfg.key_points.pick_points,
            keypoint_cfg.key_points.lift_up,
            keypoint_cfg.key_points.fling_forwards,
            keypoint_cfg.key_points.reset,
            callbacks,
        )

    @ti.func
    def _cross2d_func(self, a: ti.math.vec2, b: ti.math.vec2, c: ti.math.vec2) -> float:
        ab = b - a
        ac = c - a
        return ab[0] * ac[1] - ab[1] * ac[0]
    
    @ti.func
    def _ij2xy_func(self, i: int, j: int, dx: float, dy: float, xmin: float, ymin: float, nx: int, ny: int) -> ti.math.vec2:
        return ti.Vector([
            xmin + (j + 0.5) * dx,
            ymin + ((ny - 1 - i) + 0.5) * dy,
        ], dt=float)

    @ti.func
    def _xy2ij_func(self, x: int, y: int, dx: float, dy: float, xmin: float, ymin: float, nx: int, ny: int) -> ti.math.ivec2:
        return ti.Vector([
            ny - 1 - ti.cast(ti.math.clamp((y - ymin) / dy, 0, ny - 1), int),
            ti.cast(ti.math.clamp((x - xmin) / dx, 0, nx - 1), int),
        ], dt=int)
    
    @ti.kernel
    def _rasterize_kernel(
        self,
        grid: ti.types.ndarray(dtype=int),
        face: ti.types.ndarray(dtype=ti.math.ivec3),
        vert: ti.types.ndarray(dtype=ti.math.vec3),
        dx: float, dy: float, xmin: float, ymin: float, nx: int, ny: int, eps: float,
    ):
        """
        args:
            - grid: [B, NY, NX]
            - face: [F][3]
            - vert: [B, V][3]
        """
        for batch_idx, i, j in grid:
            grid[batch_idx, i, j] = 0
        
        for batch_idx, fid in ti.ndrange(vert.shape[0], face.shape[0]):
            a = vert[batch_idx, face[fid][0]][:2]
            b = vert[batch_idx, face[fid][1]][:2]
            c = vert[batch_idx, face[fid][2]][:2]
            i2, j1 = self._xy2ij_func(*(ti.min(a, b, c)), dx, dy, xmin, ymin, nx, ny)
            i1, j2 = self._xy2ij_func(*(ti.max(a, b, c)), dx, dy, xmin, ymin, nx, ny)
            for i in range(i1, i2 + 1):
                for j in range(j1, j2 + 1):
                    # print(f"{fid} [{i1} {i2}] X [{j1} {j2}]")
                    r = self._ij2xy_func(i, j, dx, dy, xmin, ymin, nx, ny)
                    abc = self._cross2d_func(a, b, c)
                    if abc >= 0:
                        abc = ti.max(abc, eps)
                    else:
                        abc = ti.min(abc, -eps)
                    if (
                        self._cross2d_func(a, b, r) / abc > 0. and
                        self._cross2d_func(b, c, r) / abc > 0. and
                        self._cross2d_func(c, a, r) / abc > 0.
                    ):
                        grid[batch_idx, i, j] = 1

    def calculate_coverage(self):
        self._rasterize_kernel(
            self._coverage_grid,
            self._garment.get_f2v(),
            self._garment.get_pos(),
            self._reward_coverage_cfg.dx, 
            self._reward_coverage_cfg.dy, 
            self._reward_coverage_cfg.xmin, 
            self._reward_coverage_cfg.ymin, 
            self._reward_coverage_cfg.nx, 
            self._reward_coverage_cfg.ny, 
            self._garment.dx_eps,
        )
        return utils.torch_to_numpy((
            self._coverage_grid.view(self.batch_size, -1).sum(dim=-1) *
            self._reward_coverage_cfg.dx * self._reward_coverage_cfg.dy
        ) / self._garment_rest_area)
    
    def _generate_random_rotate_z(self, sample_num: int):
        flip_y = torch.randint(low=0, high=2, size=(sample_num, ), device=self.device, dtype=self.dtype_int)
        rotate_z = torch.rand(sample_num, dtype=self.dtype, device=self.device) * (2 * math.pi) - math.pi

        rotate_z_mat = torch.zeros((sample_num, 3, 3), dtype=self.dtype, device=self.device) # [S, 3, 3]
        so3.axis_angle_to_matrix_kernel(torch.tensor([[0., 0., 1.]], dtype=self.dtype, device=self.device) * rotate_z[:, None], rotate_z_mat)

        flip_y_mat = torch.eye(3, dtype=self.dtype, device=self.device).repeat((sample_num, 1, 1))
        flip_y_mat[torch.where(flip_y)[0], :, :] = torch.tensor([
            [-1., 0., 0.],
            [0., +1., 0.],
            [0., 0., -1.],
        ], dtype=self.dtype, device=self.device)
        return flip_y_mat @ rotate_z_mat, flip_y, rotate_z
    
    def _get_best_rot(self, x: torch.Tensor, t: torch.Tensor, sample_num: int):
        """
        Args:
            x: [B, K, 3], current keypoints' position
            t: [K, 3], target
        """
        B, K, _ = x.shape
        K_, __ = t.shape
        assert K == K_ and _ == __ == 3, f"{x.shape} {t.shape}"

        mat, flip_y, rotate_z = self._generate_random_rotate_z(sample_num)# [S, 3, 3], [S], [S]

        x_rotated = (
            mat[None, None, :, :, :] @ (x - t.mean(dim=0))[:, :, None, :, None]
        )[..., 0] + t.mean(dim=0) # [B, K, S, 3] rotate around AVG(target)
        assert x_rotated.shape == (B, K, sample_num, 3)

        dx_rotated = (t[None, :, None, :] - x_rotated).mean(dim=1) # [B, S, 3]
        assert dx_rotated.shape == (B, sample_num, 3)

        dx = (mat[None, :, :, :].transpose(2, 3) @ dx_rotated[:, :, :, None])[..., 0]
        assert dx.shape == (B, sample_num, 3)

        loss3d = 0.5 * ((t[None, :, None, :] - x_rotated - dx_rotated[:, None, :, :]) ** 2) # [B, K, S, 3]
        loss = loss3d[:, :, :, :2]
        loss = loss.sum(dim=3).mean(dim=1) # [B, S]
        best_idx = loss.argmin(dim=-1) # [B]

        ret = dict(
            flip_y=utils.torch_to_numpy(flip_y[best_idx]),
            rotate_z=utils.torch_to_numpy(rotate_z[best_idx]),
            translation=utils.torch_to_numpy(dx[self._B_idx, best_idx, :]),
            loss=utils.torch_to_numpy(loss[self._B_idx, best_idx]),
            mat=utils.torch_to_numpy(mat[best_idx, :, :]),
        )
        return ret
    
    def _get_best_rot_iter(self, x: torch.Tensor, t: torch.Tensor, sample_num: int, iter_num: int):
        best = self._get_best_rot(x, t, sample_num)
        for i in range(iter_num - 1):
            curr = self._get_best_rot(x, t, sample_num)
            update_idx = np.where(curr["loss"] < best["loss"])[0]
            for k in best.keys():
                best[k][update_idx, ...] = curr[k][update_idx, ...]
        
        # result for visualize
        mat_np = best["mat"] # [B, 3, 3]
        tra_np = best["translation"] # [B, 3]
        x_np = utils.torch_to_numpy(x) # [B, K, 3]
        t_np = utils.torch_to_numpy(t) # [K, 3]
        x_tf_np = (x_np  + tra_np[:, None, :] - t_np.mean(axis=0)) @ mat_np.swapaxes(1, 2) + t_np.mean(axis=0) # [B, K, 3]
        info = dict(x=x_np, t=t_np, x_tf=x_tf_np)

        return best, info
    
    def _calculate_orientation(self, sample_num=int(1e4), iter_num=int(1e2)) -> Tuple[Dict[str, np.ndarray]]:
        """
        Return:
            angle: [B, ]
            axis: [B, 3]
            translation: [B, 3]
            loss: [B, ]
        """
        return self._get_best_rot_iter(
            self._garment.get_pos()[:, self._keypoints_vids],
            self._keypoints_target,
            sample_num,
            iter_num,
        )
    
    def get_score(self, save_orientation_path: str=None) -> Dict[str, list]:
        orientation_dict, vis_info = self._calculate_orientation()
        if isinstance(save_orientation_path, str): 
            os.makedirs(save_orientation_path, exist_ok=True)
            for batch_idx in range(self.batch_size):
                x, y, z = vis_info["x"][batch_idx, :], vis_info["t"], vis_info["x_tf"][batch_idx, :]
                trimesh.PointCloud(
                    vertices=np.concatenate([x, y, z], axis=0),
                    colors=np.array([[1., 0., 0.]] * x.shape[0] + [[0., 1., 0.]] * y.shape[0] + [[0., 0., 1.]] * z.shape[0])
                ).export(f"{save_orientation_path}/{utils.format_int(batch_idx, self.batch_size - 1)}.ply")
        return dict(
            coverage=self.calculate_coverage().tolist(),
            orientation={k: v.tolist() for k, v in orientation_dict.items()}
        )
    
    def calculate_left_right_mask(self, export_path: Optional[str]=None):
        """[B, H, W, 2]"""
        table_height_np = utils.torch_to_numpy(self._sim_env.get_table_height())
        mesh_rest = self._sim_env.garment_rest_mesh

        vert = self._sim_env.garment.get_pos()
        vert_rest = torch.tensor(mesh_rest.vertices, dtype=self.dtype, device=self.device)
        vid_lr = np.array([self._sim_env.garment_keypoints[k] for k in ["upper_left", "upper_right"]], dtype=int)

        xy_range_l = []
        reproject_camera_prop = self._agent.get_reproject_camera_prop()
        reproject_camera_pose = self._agent.get_reproject_camera_pose()
        camera_info = policy_utils.CameraInfo(asdict(reproject_camera_prop), reproject_camera_pose)
        H, W = reproject_camera_prop.height, reproject_camera_prop.width
        for batch_idx in range(self.batch_size):
            xmin, ymax, _ = policy_utils.ijd2xyz(-0.5, -0.5, reproject_camera_pose[2] - table_height_np[batch_idx], camera_info)
            xmax, ymin, _ = policy_utils.ijd2xyz(H - 0.5, W - 0.5, reproject_camera_pose[2] - table_height_np[batch_idx], camera_info)
            xy_range_l.append([[xmin, ymin], [xmax, ymax]])
        ans = torch.zeros((self.batch_size, H, W, 2), dtype=self.dtype, device=self.device)
        grasp_wrong = ans.clone()

        _calculate_atractable_keypoints_kernel(
            vert, vert_rest,
            vid_lr, torch.tensor(xy_range_l, dtype=self.dtype, device=self.device),
            ans, grasp_wrong,
            float(self._cfg.reward.keypoints.gripper_radius), 
            float(self._cfg.reward.keypoints.dist_threshold),
        )

        if isinstance(export_path, str):
            ans_np = utils.torch_to_numpy(ans)
            for lr_idx, lr_str in enumerate(["left", "right"]):
                for batch_idx in range(self.batch_size):
                    file_path = os.path.join(export_path, lr_str, utils.format_int(batch_idx, self.batch_size - 1) + ".npy")
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    np.save(file_path, ans_np[batch_idx, :, :, lr_idx])
        
        return ans