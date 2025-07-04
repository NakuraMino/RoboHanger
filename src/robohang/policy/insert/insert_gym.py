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
from robohang.policy.gym_wrapper import BaseGym


@ti.data_oriented
class InsertGym(BaseGym):
    def __init__(self, sim_env: SimEnv, agent: BaseAgent, gym_cfg: omegaconf.DictConfig) -> None:
        super().__init__(sim_env, agent, gym_cfg)

        # primitive hyperparameters
        self._parameter: omegaconf.DictConfig = copy.deepcopy(self._cfg.parameter)
        self._init_garment_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.init_garment)
        self._lift_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.lift)
        self._press_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.press)
        self._drag_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.drag)
        self._rotate_cfg: omegaconf.DictConfig = copy.deepcopy(self._cfg.primitive.rotate)
 
        # init
        self._left_vid = self._sim_env.garment_keypoints[self._parameter.keypoints_name.left]
        self._right_vid = self._sim_env.garment_keypoints[self._parameter.keypoints_name.right]
        self._init_garment_pick_vids_l = np.where(
            np.linalg.norm((
                self._sim_env.garment_rest_mesh.vertices - 
                self._sim_env.garment_rest_mesh.vertices[self._left_vid] - 
                np.array(self._init_garment_cfg.pick_place_offset.left)
            ), axis=1) < self._init_garment_cfg.pick_place_radius
        )[0]
        self._init_garment_pick_vids_r = np.where(
            np.linalg.norm((
                self._sim_env.garment_rest_mesh.vertices - 
                self._sim_env.garment_rest_mesh.vertices[self._right_vid] - 
                np.array(self._init_garment_cfg.pick_place_offset.right)
            ), axis=1) < self._init_garment_cfg.pick_place_radius
        )[0]

        self._check_hanger_rpy()

        # reward
        pcd = trimesh.load(utils.get_path_handler()(self._cfg.parameter.fps_on_unit_sphere))
        self._fps_on_unit_sphere = torch.tensor(pcd.vertices, dtype=self.dtype, device=self.device)
        """[S, 3]"""
        self._solid_angle = torch.zeros((self.batch_size, ), dtype=self.dtype, device=self.device)
        """float, [B, ]"""
        self._sample_hit = torch.zeros((self.batch_size, self._fps_on_unit_sphere.shape[0]), dtype=self.dtype_int, device=self.device)
        """int, [B, S]"""

        # state
        self._init_state = self._sim_env.get_state()
    
    def _damp_callback(self, *args, **kwargs):
        vel = self._garment.get_vel()
        vel *= (1. - self._parameter.damp_per_substep)
        self._garment.set_vel(vel)

    def _endpoint_xyz_to_hanger_xyz(
        self,
        endpoint_name: Literal["left", "right"],
        endpoint_xyz: torch.Tensor,
        hanger_rpy: np.ndarray,
    ):
        """
        endpoint_xyz: world frame endpoint's xyz
        hanger_rpy: world frame hanger's rpy
        """
        rot_mat = tra.euler_matrix(*hanger_rpy)[:3, :3]
        xyz_hanger_frame = np.array(getattr(self._sim_env.hanger_meta, endpoint_name))
        xyz_world_frame = rot_mat @ xyz_hanger_frame
        return endpoint_xyz - torch.tensor(xyz_world_frame, dtype=self.dtype, device=self.device)

    def _check_hanger_rpy(self):
        assert np.allclose(
            tra.euler_matrix(*(self._agent.action_rpy.press.press.right)) @
            tra.euler_matrix(*(self._press_cfg.hanger_rpy.transform)),
            tra.euler_matrix(*(self._press_cfg.hanger_rpy.absolute)),
        ), f"press rpy is not correct"

        assert np.allclose(
            tra.euler_matrix(*(self._agent.action_rpy.drag.regrasp.left)) @
            tra.euler_matrix(*(self._drag_cfg.hanger_rpy.transform)),
            tra.euler_matrix(*(self._drag_cfg.hanger_rpy.absolute)),
        ), f"drag rpy is not correct"

        assert np.allclose(
            tra.euler_matrix(*(self._agent.action_rpy.rotate.rotate.left)) @
            tra.euler_matrix(*(self._rotate_cfg.hanger_rpy.transform)),
            tra.euler_matrix(*(self._rotate_cfg.hanger_rpy.absolute)),
        ), f"rotate rpy is not correct"

    @ti.kernel
    def _generate_perturb_garment_kernel(
        self,
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        global_lin_vel: ti.types.ndarray(dtype=ti.math.vec3),
        global_ang_vel: ti.types.ndarray(dtype=ti.math.vec3),
        global_rot_xyz: ti.types.ndarray(dtype=ti.math.vec3),
        local_vel: ti.types.ndarray(dtype=ti.math.vec3),
        perturb: ti.types.ndarray(dtype=ti.math.vec3),
    ):
        """
        Args:
            - pos: float, [B, V, 3]
            - global_lin_vel: float, [B, 3]
            - global_ang_vel: float, [B, 3]
            - global_rot_xyz: float, [B, 3]
            - local_vel: float, [B, V, 3]
            - perturb: float, [B, V, 3]
        """
        for batch_idx, vid in ti.ndrange(perturb.shape[0], perturb.shape[1]):
            perturb[batch_idx, vid] = (
                global_lin_vel[batch_idx] + # global linear velocity
                global_ang_vel[batch_idx].cross(pos[batch_idx, vid] - global_rot_xyz[batch_idx]) + # global angular velocity
                local_vel[batch_idx, vid] # local velocity
            )

    def _generate_perturb_garment(
        self,
        global_lin_vel_dict: Dict[Literal["mean", "std"], Tuple[float, float, float]],
        global_ang_vel_dict: Dict[Literal["mean", "std"], Tuple[float, float, float]],
        global_rot_xyz_dict: Dict[Literal["mean", "std"], Tuple[float, float, float]],
        local_vel_std: Tuple[float, float, float]
    ) -> Tuple[torch.Tensor, Dict[Literal["global_lin_vel", "global_ang_vel", "global_rot_xyz"], torch.Tensor]]:
        def randn(mean, std, size):
            return (
                torch.randn(size, dtype=self.dtype, device=self.device) * 
                torch.tensor(std, dtype=self.dtype, device=self.device) +
                torch.tensor(mean, dtype=self.dtype, device=self.device)
            )
        global_lin_vel = randn(global_lin_vel_dict["mean"], global_lin_vel_dict["std"], (self.batch_size, 3))
        global_ang_vel = randn(global_ang_vel_dict["mean"], global_ang_vel_dict["std"], (self.batch_size, 3))
        global_rot_xyz = randn(global_rot_xyz_dict["mean"], global_rot_xyz_dict["std"], (self.batch_size, 3))
        local_vel = randn((0., 0., 0.), local_vel_std, (self.batch_size, self._garment.nv, 3))
        perturb = torch.zeros_like(local_vel)

        self._generate_perturb_garment_kernel(
            self._garment.get_pos(),
            global_lin_vel,
            global_ang_vel,
            global_rot_xyz,
            local_vel,
            perturb,
        )

        info = dict(
            global_lin_vel=global_lin_vel,
            global_ang_vel=global_ang_vel,
            global_rot_xyz=global_rot_xyz,
        )
        return perturb, info

    def _drag_garment(
        self, 
        xyz_s_list: List[torch.Tensor], 
        xyz_e_list: List[torch.Tensor], 
        vids_list: List[torch.Tensor], 
        stiffness: float, 
        step: int, 
        callbacks
    ):
        logger.info(f"InsertGym drag_garment\nxyz_s:\n{xyz_s_list}\nxyz_e:\n{xyz_e_list}\nvids:\n{vids_list}")
        self._drag_callback_counter = 0
        total_substeps = step * self._sim_env.sim.substeps

        def drag_callback(*args, **kwargs):
            self._drag_callback_counter += 1

            pos = self._garment.get_pos()
            external_force = self._garment.get_external_force()
            external_hessian = self._garment.get_external_hessian()

            for xyz_s, xyz_e, vids in zip(xyz_s_list, xyz_e_list, vids_list):
                err = -pos[self._B_idx, vids] + (
                    xyz_s + (xyz_e - xyz_s) * self._drag_callback_counter / total_substeps
                ) # [B, 3]
                external_force[self._B_idx, vids] += err * stiffness
                external_hessian[self._B_idx, vids] += stiffness * torch.eye(3, dtype=self.dtype, device=self.device)
                logger.info(f"InsertGym drag_callback avg_err:{float(torch.mean(torch.norm(err, dim=1)))}")

            self._garment.set_external_force(external_force)
            self._garment.set_external_hessian(external_hessian)

        self._sim_env.simulate(step, callbacks=[drag_callback] + callbacks)
        
        con = self._garment.get_constraint() 
        con[...] = 1.0
        self._garment.set_constraint(con)
    
    def _random_garment_drag_inter_xyz(self, drag_inter_cfg: omegaconf.DictConfig, dist_lr: torch.Tensor):
        xc = utils.map_01_ab(torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), drag_inter_cfg.x[0], drag_inter_cfg.x[1])
        yc = utils.map_01_ab(torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), drag_inter_cfg.y[0], drag_inter_cfg.y[1])
        zc = utils.map_01_ab(torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), drag_inter_cfg.h[0], drag_inter_cfg.h[1]) + self._sim_env.get_table_height()
        xyzl = torch.concat([(xc - dist_lr / 2)[:, None], yc[:, None], zc[:, None]], dim=1)
        xyzr = torch.concat([(xc + dist_lr / 2)[:, None], yc[:, None], zc[:, None]], dim=1)
        logger.info(f"random garment drag xy:\nxyzl:{xyzl}\nxyzr:{xyzr}")
        return xyzl, xyzr

    def primitive_init_garment(self, callbacks: Optional[List[Callable]]=None):
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        # set init xyz
        pos = self._garment.get_pos()
        pos[...] = torch.tensor(self._sim_env.garment_rest_mesh.vertices, dtype=self.dtype, device=self.device)
        self._garment.set_pos(pos) # to rest shape

        perturb, info = self._generate_perturb_garment(**(self._init_garment_cfg.perturb))
        logger.info(f"perturb:\n{info}")
        self._garment.set_vel(self._garment.get_vel() + perturb)

        random_h = utils.map_01_ab(torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), *(self._init_garment_cfg.init_pos.h))
        random_y = utils.map_01_ab(torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), *(self._init_garment_cfg.init_pos.y))
        logger.info(f"random_h:\n{random_h}\nrandom_y:{random_y}")
        pos = self._garment.get_pos() # [B, V, 3]
        pos[...] = torch.tensor(self._sim_env.garment_rest_mesh.vertices, dtype=self.dtype, device=self.device)
        pos[:, :, 1] += random_y[:, None]
        pos[:, :, 2] += (self._sim_env.get_table_height() + random_h)[:, None]
        self._garment.set_pos(pos)

        # random drag
        vids_l = torch.tensor(np.random.choice(self._init_garment_pick_vids_l, size=(self.batch_size, )), dtype=torch.long, device=self.device)
        vids_r = torch.tensor(np.random.choice(self._init_garment_pick_vids_r, size=(self.batch_size, )), dtype=torch.long, device=self.device)
        xyz_0_l = self._garment.get_pos()[self._B_idx, vids_l]
        xyz_0_r = self._garment.get_pos()[self._B_idx, vids_r]
        dist_lr = torch.norm(xyz_0_l - xyz_0_r, dim=1) * utils.map_01_ab(
            torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), 
            self._init_garment_cfg.drag_hand_dist[0], self._init_garment_cfg.drag_hand_dist[1]
        ) # [B, ]

        self._drag_garment(
            [xyz_0_l, xyz_0_r], 
            [xyz_0_l, xyz_0_r], 
            [vids_l, vids_r], 
            self._parameter.stiffness, 
            self._init_garment_cfg.init_pos.steps, 
            callbacks=[self._damp_callback] + callbacks
        )

        xyz_l_prev, xyz_r_prev = xyz_0_l, xyz_0_r
        self._sim_env.set_substep("accurate")
        for drag_inter_cfg in self._init_garment_cfg.drag_inter:
            xyz_l_curr, xyz_r_curr = self._random_garment_drag_inter_xyz(drag_inter_cfg, dist_lr)
            self._drag_garment(
                [xyz_l_prev, xyz_r_prev], 
                [xyz_l_curr, xyz_r_curr], 
                [vids_l, vids_r], 
                self._parameter.stiffness, 
                drag_inter_cfg.steps, 
                callbacks=callbacks
            )
            xyz_l_prev, xyz_r_prev = xyz_l_curr, xyz_r_curr
        self._sim_env.set_substep("efficient")

        # wait to fall
        self._sim_env.simulate(self._init_garment_cfg.wait_step, callbacks=callbacks)

    def depre_primitive_lift(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_lift raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "lift"

        # preprocess input
        xyz = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._lift_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._lift_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz[:, 2] = self._sim_env.get_table_height()

        # pick
        self._primitive_move_to_pick_points(
            self._lift_cfg.pick_points,
            primitive_name,
            ["left"],
            callbacks,
            xyz_l=xyz,
        )
    
    def depre_primitive_press(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_press raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "press"
        action_name = "press"

        # preprocess input
        xyz_left_end = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._press_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._press_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz_left_end[:, 2] = self._sim_env.get_table_height()
        xyz = self._endpoint_xyz_to_hanger_xyz(
            "left", xyz_left_end,
            np.array(self._press_cfg.hanger_rpy.absolute),
        )
        logger.info(f"primitive_press xyz:\n{xyz}")

        # fix hanger
        self._sim_env.robot_hanger.set_mode(
            "Fix", self._agent.right_grasp_link, 
            torch.tensor(tra.euler_matrix(*(self._press_cfg.hanger_rpy.transform))),
        )

        # move to xyz_left_end
        self._sim_env.penetration_checker.set_check_gripper_hanger(True, False)
        for h_idx, h_str in enumerate(["h_upper", "h_inter", "h_lower"]):
            xyz_curr = xyz + torch.tensor(
                [0., 0., getattr(self._press_cfg.press, h_str)], 
                dtype=self.dtype, device=self.device
            )
            info = self._set_gripper_target_wrap(
                xyz_curr, primitive_name, action_name, "right", 
                use_ik_init_cfg=bool(h_idx == 0), xyz_c=xyz, 
            )
            logger.info(f"{primitive_name} {action_name} {h_str}:\n{info}")
            self._sim_env.set_actor_speed("interp", steps=self._press_cfg.press.steps[h_idx])
            self._sim_env.simulate(self._press_cfg.press.steps[h_idx], callbacks=callbacks)
        self._sim_env.penetration_checker.set_check_gripper_hanger(False, False)
        
        # insert
        xyz_final = xyz_curr.clone()
        xyz_final[:, :2] += torch.tensor(self._press_cfg.insert.xy, dtype=self.dtype, device=self.device)

        # insert step 1
        xyz_r = (xyz_curr + xyz_final) / 2
        info = self._set_gripper_target_wrap(
            xyz_r, primitive_name, action_name, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, 
        )
        logger.info(f"{primitive_name} {action_name} insert:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[0])
        self._sim_env.simulate(self._press_cfg.insert.steps[0], callbacks=callbacks)

        # move away left hand
        self._sim_env.set_substep("accurate")
        self._release_grippers()
        xyz_l = (
            self._sim_env.robot.get_link_pos(self._agent.left_grasp_link)[:, :3] + 
            torch.tensor(self._press_cfg.insert.xyz_delta_left, dtype=self.dtype, device=self.device)   
        )
        info = self._set_gripper_target_wrap(
            xyz_l, primitive_name, action_name, "left", 
            use_ik_init_cfg=False, xyz_c=xyz_l, 
        )
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[1])
        self._sim_env.simulate(self._press_cfg.insert.steps[1], callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        # insert step 2
        xyz_r = xyz_final
        info = self._set_gripper_target_wrap(
            xyz_r, primitive_name, action_name, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, 
        )
        logger.info(f"{primitive_name} {action_name} insert:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[2])
        self._sim_env.simulate(self._press_cfg.insert.steps[2], callbacks=callbacks)

        # reset
        self._sim_env.robot_hanger.set_mode("Release")
        self._primitive_reset_grippers(
            self._press_cfg.reset,
            primitive_name,
            ["left", "right"],
            callbacks,
            xyz_l=self._sim_env.robot.get_link_pos(self._agent.left_grasp_link)[:, :3],
            xyz_r=self._sim_env.robot.get_link_pos(self._agent.right_grasp_link)[:, :3],
        )
    
    def primitive_press(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_press raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "press"
        action_name = "press"

        # preprocess input
        xyz_left_end = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._press_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._press_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz_left_end[:, 2] = self._sim_env.get_table_height()
        xyz = self._endpoint_xyz_to_hanger_xyz(
            "left", xyz_left_end,
            np.array(self._press_cfg.hanger_rpy.absolute),
        )
        logger.info(f"primitive_press xyz:\n{xyz}")

        # fix hanger
        self._sim_env.robot_hanger.set_mode(
            "Fix", self._agent.right_grasp_link, 
            torch.tensor(tra.euler_matrix(*(self._press_cfg.hanger_rpy.transform))),
        )

        # move to xyz_left_end
        for h_idx, h_str in enumerate(["h_upper", "h_inter", "h_lower"]):
            xyz_curr = xyz + torch.tensor(
                [0., 0., getattr(self._press_cfg.press, h_str)], 
                dtype=self.dtype, device=self.device
            )
            info = self._set_gripper_target_wrap(
                xyz_curr, primitive_name, action_name, "right", 
                use_ik_init_cfg=bool(h_idx == 0), xyz_c=xyz, 
            )
            logger.info(f"{primitive_name} {action_name} {h_str}:\n{info}")
            self._sim_env.set_actor_speed("interp", steps=self._press_cfg.press.steps[h_idx])
            self._sim_env.simulate(self._press_cfg.press.steps[h_idx], callbacks=callbacks)

    def primitive_lift(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_lift raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "lift"

        # preprocess input
        xyz = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._lift_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._lift_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz[:, 2] = self._sim_env.get_table_height()

        # pick
        self._sim_env.penetration_checker.set_check_gripper_hanger(True, False)
        self._primitive_move_to_pick_points(
            self._lift_cfg.pick_points,
            primitive_name,
            ["left"],
            callbacks,
            xyz_l=xyz,
        )
        self._sim_env.penetration_checker.set_check_gripper_hanger(False, False)

        # deprecate insert actions
        # the primitive_name and action_name use the old names
        
        # insert
        primitive_name = "press"
        action_name = "press"
        xyz_curr = self._sim_env.grippers["right"].get_rigid().get_pos()[:, :3]
        xyz_final = xyz_curr.clone()
        xyz_final[:, :2] += torch.tensor(self._press_cfg.insert.xy, dtype=self.dtype, device=self.device)

        # insert step 1
        xyz_r = (xyz_curr + xyz_final) / 2
        info = self._set_gripper_target_wrap(
            xyz_r, primitive_name, action_name, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, 
        )
        logger.info(f"{primitive_name} {action_name} insert:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[0])
        self._sim_env.simulate(self._press_cfg.insert.steps[0], callbacks=callbacks)

        # move away left hand
        self._sim_env.set_substep("accurate")
        self._release_grippers()
        xyz_l = (
            self._sim_env.robot.get_link_pos(self._agent.left_grasp_link)[:, :3] + 
            torch.tensor(self._press_cfg.insert.xyz_delta_left, dtype=self.dtype, device=self.device)   
        )
        info = self._set_gripper_target_wrap(
            xyz_l, primitive_name, action_name, "left", 
            use_ik_init_cfg=False, xyz_c=xyz_l, 
        )
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[1])
        self._sim_env.simulate(self._press_cfg.insert.steps[1], callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        # insert step 2
        xyz_r = xyz_final
        info = self._set_gripper_target_wrap(
            xyz_r, primitive_name, action_name, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, 
        )
        logger.info(f"{primitive_name} {action_name} insert:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._press_cfg.insert.steps[2])
        self._sim_env.simulate(self._press_cfg.insert.steps[2], callbacks=callbacks)

        # reset
        self._sim_env.robot_hanger.set_mode("Release")
        self._primitive_reset_grippers(
            self._press_cfg.reset,
            primitive_name,
            ["left", "right"],
            callbacks,
            xyz_l=self._sim_env.robot.get_link_pos(self._agent.left_grasp_link)[:, :3],
            xyz_r=self._sim_env.robot.get_link_pos(self._agent.right_grasp_link)[:, :3],
        )
    
    def primitive_drag(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._release_grippers()
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_drag raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "drag"

        # preprocess input
        xyz = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._drag_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._drag_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz[:, 2] = self._sim_env.get_table_height()
        
        # regrasp
        xyz_l = self._sim_env.hanger.get_pos()[:, :3]
        info = self._set_gripper_target_wrap(
            xyz_l, primitive_name, "regrasp", "left", 
            use_ik_init_cfg=True, xyz_c=xyz_l, 
        )
        logger.info(f"{primitive_name} regrasp left:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._drag_cfg.regrasp.steps[0])
        self._sim_env.simulate(self._drag_cfg.regrasp.steps[0], callbacks=callbacks)

        self._sim_env.robot_hanger.set_mode(
            "Fix", self._agent.left_grasp_link, 
            torch.tensor(tra.euler_matrix(*(self._drag_cfg.hanger_rpy.transform))),
        )

        # slightly lift the hanger
        xyz_l[:, 2] = self._drag_cfg.regrasp.h_hanger_upper + self._sim_env.get_table_height()
        info = self._set_gripper_target_wrap(
            xyz_l, primitive_name, "regrasp", "left", 
            use_ik_init_cfg=False, xyz_c=xyz_l, 
        )
        logger.info(f"{primitive_name} regrasp lift:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._drag_cfg.regrasp.steps[1])
        self._sim_env.simulate(self._drag_cfg.regrasp.steps[1], callbacks=callbacks)

        # pick
        self._sim_env.penetration_checker.set_check_gripper_hanger(False, True)
        self._primitive_move_to_pick_points(
            self._drag_cfg.pick_points,
            primitive_name,
            ["right"],
            callbacks,
            xyz_r=xyz,
        )
        self._sim_env.penetration_checker.set_check_gripper_hanger(False, False)

        # drag
        for drag_step_cfg in self._drag_cfg.drag:
            xyz_r = self._get_current_xyz("right")
            xyz_r[:, 2] = self._sim_env.get_table_height() + self._drag_cfg.pick_points.h_later
            xyz_r[:, :2] += torch.tensor(drag_step_cfg.xy, dtype=self.dtype, device=self.device)
            info = self._set_gripper_target_wrap(
                xyz_r, primitive_name, "drag", "right", 
                use_ik_init_cfg=False, xyz_c=xyz_r, 
            )
            logger.info(f"{primitive_name} drag drag:\n{info}")
            self._sim_env.set_actor_speed("interp", steps=drag_step_cfg.steps)
            self._sim_env.simulate(drag_step_cfg.steps, callbacks=callbacks)
    
    def primitive_rotate(
        self,
        xy: torch.Tensor,
        callbacks: Optional[List[Callable]]=None,
    ):
        """xy: [B, 2]"""
        if callbacks is None:
            callbacks = []
        self._sim_env.set_substep("efficient")

        logger.info(f"primitive_rotate raw:\n{xy}")
        assert xy.shape == (self.batch_size, 2)

        primitive_name = "rotate"

        # preprocess input
        xyz_right_end = F.pad(torch.clip(
            xy.to(dtype=self.dtype).to(device=self.device),
            min=torch.tensor([self._rotate_cfg.action_space.min], device=self.device),
            max=torch.tensor([self._rotate_cfg.action_space.max], device=self.device),
        ), (0, 1), "constant", 0)
        xyz_right_end[:, 2] = self._sim_env.get_table_height() + float(self._rotate_cfg.rotate.h)
        xyz = self._endpoint_xyz_to_hanger_xyz(
            "right", xyz_right_end,
            np.array(self._rotate_cfg.hanger_rpy.absolute),
        )
        logger.info(f"primitive_press xyz:\n{xyz}")

        # rotate
        info = self._set_gripper_target_wrap(
            xyz, primitive_name, "rotate", "left", 
            use_ik_init_cfg=False, xyz_c=xyz, 
        )
        logger.info(f"{primitive_name} rotate rotate:\n{info}")
        self._sim_env.set_actor_speed("interp", steps=self._rotate_cfg.rotate.steps[0])
        self._sim_env.simulate(self._rotate_cfg.rotate.steps[0], callbacks=callbacks)
        self._release_grippers()

        # liftup
        self._sim_env.set_substep("accurate")
        for liftup_str in ["liftup1", "liftup2"]:
            xyz_l = torch.tensor(getattr(self._rotate_cfg, liftup_str).xyh_l, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1)
            xyz_l[:, 2] += self._sim_env.get_table_height()
            info = self._set_gripper_target_wrap(
                xyz_l, primitive_name, liftup_str, "left", 
                use_ik_init_cfg=False, xyz_c=xyz_l
            )
            logger.info(f"{primitive_name} rotate {liftup_str}:\n{info}")
            self._sim_env.set_actor_speed("interp", steps=getattr(self._rotate_cfg, liftup_str).steps)
            self._sim_env.simulate(getattr(self._rotate_cfg, liftup_str).steps, callbacks=callbacks)
        self._sim_env.set_substep("efficient")

        # reset
        self._sim_env.robot_hanger.set_mode("Release")
        self._primitive_reset_grippers(
            self._rotate_cfg.reset,
            primitive_name,
            ["left", "right"],
            callbacks,
            xyz_l=self._sim_env.robot.get_link_pos(self._agent.left_grasp_link)[:, :3],
            xyz_r=self._sim_env.robot.get_link_pos(self._agent.right_grasp_link)[:, :3],
        )

    def _generate_random_action(self, num: int, action_space_cfg: omegaconf.DictConfig) -> Dict[Literal["xy"], torch.Tensor]:
        return dict(
            xy=torch.concat([
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
    
    def random_lift_action(self, num: Optional[int]=None):
        if num is None:
            num = self.batch_size
        action = self._generate_random_action(int(num), self._lift_cfg.action_space)
        return action
    
    def random_press_action(self, num: Optional[int]=None):
        if num is None:
            num = self.batch_size
        action = self._generate_random_action(int(num), self._press_cfg.action_space)
        return action
    
    def random_drag_action(self, num: Optional[int]=None):
        if num is None:
            num = self.batch_size
        action = self._generate_random_action(int(num), self._drag_cfg.action_space)
        return action
    
    def random_rotate_action(self, num: Optional[int]=None):
        if num is None:
            num = self.batch_size
        action = self._generate_random_action(int(num), self._rotate_cfg.action_space)
        return action
    
    @ti.kernel
    def _calculate_solid_angle_kernel(
        self, 
        fps_on_unit_sphere: ti.types.ndarray(dtype=ti.math.vec3),
        sample_hit: ti.types.ndarray(dtype=int),
        solid_angle: ti.types.ndarray(dtype=float),
        center_xyz: ti.types.ndarray(dtype=ti.math.vec3),
        pos: ti.types.ndarray(dtype=ti.math.vec3),
        f2v: ti.types.ndarray(dtype=ti.math.ivec3),
        dx_eps: float,
    ):
        """
        Args:
            - fps_on_unit_sphere: float, [S][3]
            - sample_hit: int, [B, S]
            - solid_angle: float, [B]
            - center_xyz: float, [B][3]
            - pos: float, [B, V][3]
            - f2v: int, [F][3]
        """
        B = sample_hit.shape[0]
        S = sample_hit.shape[1]
        F = f2v.shape[0]
        for batch_idx, sample_idx in ti.ndrange(B, S):
            sample_hit[batch_idx, sample_idx] = 0

        for batch_idx, sample_idx, fid in ti.ndrange(B, S, F):
            p0 = ti.cast(pos[batch_idx, f2v[fid][0]], ti.f64)
            p1 = ti.cast(pos[batch_idx, f2v[fid][1]], ti.f64)
            p2 = ti.cast(pos[batch_idx, f2v[fid][2]], ti.f64)
            p3 = ti.cast(center_xyz[batch_idx], ti.f64)
            p4 = ti.cast(center_xyz[batch_idx] + fps_on_unit_sphere[sample_idx], ti.f64)
            
            mat = ti.Matrix.zero(ti.f64, 3, 3)
            mat[:, 0] = p3 - p4
            mat[:, 1] = p1 - p0
            mat[:, 2] = p2 - p0

            xyz_scale = ti.abs(mat).sum() / 9
            mat_det = mat.determinant()
            if ti.abs(mat_det) > (xyz_scale ** 2) * dx_eps:
                right = p3 - p0
                left = mat.inverse() @ right

                a, b, c = 1. - left[1] - left[2], left[1], left[2]
                t = left[0]
                abct = ti.Vector([a, b, c, t], ti.f64)
                abc = ti.Vector([a, b, c], ti.f64)
                zero_f64 = ti.cast(0.0, ti.f64)
                one_f64 = ti.cast(1.0, ti.f64)
                if (zero_f64 < abct).all() and (abc < one_f64).all():
                    sample_hit[batch_idx, sample_idx] = 1
        
        for batch_idx in range(B):
            solid_angle[batch_idx] = 0.

        for batch_idx, sample_idx in ti.ndrange(B, S):
            if sample_hit[batch_idx, sample_idx] != 0:
                solid_angle[batch_idx] += 4 * ti.math.pi / S

    def _calculate_solid_angle(self, center_xyz: torch.Tensor) -> torch.Tensor:
        assert self.batch_size * self._fps_on_unit_sphere.shape[0] * self._garment.nf < sim_utils.MAX_RANGE
        self._calculate_solid_angle_kernel(
            self._fps_on_unit_sphere,
            self._sample_hit,
            self._solid_angle,
            center_xyz,
            self._garment.get_pos(),
            self._garment.get_f2v(),
            self._garment.dx_eps,
        )
        return self._solid_angle.clone()
    
    def _get_current_xyz(self, endpoint_name: Literal["left", "right"]):
        """return xyz_world_frame, [B, 3]"""
        xyz_hanger_frame = torch.tensor(
            [*(getattr(self._sim_env.hanger_meta, endpoint_name)), 1.],
            dtype=self.dtype, device=self.device,
        ).repeat(self.batch_size, 1) # [B, 4]
        hanger_mat = so3.pos7d_to_matrix(self._sim_env.hanger.get_pos()) # [B, 4, 4]
        xyz_world_frame = (hanger_mat @ xyz_hanger_frame[..., None])[:, :3, 0] # [B, 3]
        return xyz_world_frame.contiguous()

    def get_current_xyz(self, endpoint_name: Literal["left", "right"]):
        return self._get_current_xyz(endpoint_name)
    
    def get_score(self) -> Dict[str, list]:
        return dict(
            left=utils.torch_to_numpy(self._calculate_solid_angle(self._get_current_xyz("left")) / (math.pi * 4)).tolist(),
            right=utils.torch_to_numpy(self._calculate_solid_angle(self._get_current_xyz("right")) / (math.pi * 4)).tolist(),
        )