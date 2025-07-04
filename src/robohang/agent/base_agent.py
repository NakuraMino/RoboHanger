import taichi as ti

import logging
logger = logging.getLogger(__name__)

from typing import Literal, Any, Tuple, Dict, Optional, List, Union
import copy
import os
import random
import pprint

import torch
import numpy as np

import trimesh
import trimesh.transformations as tra

import omegaconf

import robohang.sim.so3 as so3
import robohang.sim.sim_utils as sim_utils

from robohang.env.sim_env import SimEnv, RobotGripper
import robohang.common.utils as utils
from robohang.env.sapien_renderer import CameraProperty, reproject, camera_pose_to_matrix, camera_property_to_intrinsics_matrix, randomize_camera


@ti.data_oriented
class BaseAgent:
    def __init__(self, sim_env: SimEnv, agent_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(agent_cfg, omegaconf.DictConfig)
        assert isinstance(sim_env, SimEnv)

        # initialize
        self._cfg = copy.deepcopy(agent_cfg)
        self._sim_env = sim_env

        # constants
        self.batch_size = self._sim_env.batch_size
        self.device = self._sim_env.device
        self.dtype = self._sim_env.dtype
        self.dtype_int = self._sim_env.dtype_int

        # init robot
        self._sim_env.robot.set_forward_base_link_pos_vel(
            self._cfg.init.base.name,
            torch.tensor([self._cfg.init.base.pos] * self.batch_size, dtype=self.dtype, device=self.device),
            sim_utils.create_zero_6d_vel(self.batch_size, self.dtype, self.device),
        )
        qpos = self._sim_env.robot.get_cfg_pos()
        for joint_name, joint_cfg in qpos.items():
            joint_cfg[...] = torch.tensor([self._cfg.init.qpos[joint_name]] * self.batch_size, dtype=self.dtype, device=self.device)
        self._sim_env.robot.forward_kinematics(qpos, self._sim_env.robot_zero_qvel)
        self._sim_env.actor.set_actor_dof_targets(qpos, self._sim_env.robot_zero_qvel)
        self._init_qpos = self._sim_env.robot.get_cfg_pos()

        # camera
        self._camera_info = dict()
        for s in ["top", "left", "right", "side"]:
            camera_cfg = getattr(self._cfg.camera, s)
            if s == "side":
                self._camera_info[s] = dict(
                    pos=copy.deepcopy(camera_cfg.pos),
                )
            else:
                self._camera_info[s] = dict(
                    link=camera_cfg.link,
                    origin=tra.translation_matrix(camera_cfg.origin.xyz) @ 
                           tra.euler_matrix(*(camera_cfg.origin.rpy))
                )
        
        self._camera_prop: Dict[str, CameraProperty] = dict()
        for s in ["small", "medium", "large"]:
            self._camera_prop[s] = CameraProperty(**(getattr(self._cfg.camera.prop, s)))
        self.direct_obs = [float(x) for x in np.array(self._cfg.camera.direct.pos)]
        assert np.array(self.direct_obs).shape == (7, )

        # end effector
        self._left_grasp_link = str(self._cfg.grasp.left.link)
        self._right_grasp_link = str(self._cfg.grasp.right.link)
        self._sim_env.grippers["left"].set_link_str(self._left_grasp_link)
        self._sim_env.grippers["right"].set_link_str(self._right_grasp_link)

        # test camera
        self.get_obs("top", "small", 0, False)
        self.get_obs("left", "medium", 0, True)
        self.get_obs("right", "large", 0, False)
        self.get_obs("side", "small", 0, True)

        # constants
        self._ik_solver_cfg = copy.deepcopy(agent_cfg.inverse_kinematics.solver)
        self._ik_init: dict = copy.deepcopy(agent_cfg.inverse_kinematics.init)
        self._action_rpy: dict = copy.deepcopy(agent_cfg.action_rpy)

    @property
    def ik_init(self):
        return self._ik_init
    
    @property
    def action_rpy(self):
        return self._action_rpy

    @property
    def left_grasp_link(self):
        return self._left_grasp_link
    
    @property
    def right_grasp_link(self):
        return self._right_grasp_link
    
    @sim_utils.GLOBAL_TIMER.timer
    def get_obs(
        self, 
        camera_name: Literal["top", "left", "right", "side", "direct"], 
        image_size: Literal["small", "medium", "large"], 
        batch_idx: int,
        randomize: bool,
        **kwargs
    ):

        camera_prop = self._camera_prop[image_size]
        if camera_name in ["top", "left", "right"]:
            camera_pose_mat = utils.torch_to_numpy(
                (self._sim_env.robot.urdf.link_transform_map[self._camera_info[camera_name]["link"]])[batch_idx, :]
            ) @ self._camera_info[camera_name]["origin"]
            camera_pose = camera_pose_mat[:3, 3].tolist() + tra.quaternion_from_matrix(camera_pose_mat).tolist()
        elif camera_name == "side":
            camera_pose = self._camera_info[camera_name]["pos"]
        elif camera_name == "direct":
            camera_pose = kwargs["pos"]
        else:
            raise NotImplementedError(camera_name)
        
        if randomize:
            camera_prop, camera_pose = randomize_camera(camera_prop, camera_pose, self._cfg.camera.rand)
        result, mask_str_to_idx = self._sim_env.take_picture(camera_prop, camera_pose, batch_idx)
        camera_info: Dict[str, Union[CameraProperty, List[float]]] = dict(camera_prop=camera_prop, camera_pose=camera_pose)
        return result, mask_str_to_idx, camera_info
    
    def _get_reproject_camera_prop(self):
        return CameraProperty(**(self._cfg.policy_obs.reproject.camera_prop))
    
    def get_reproject_camera_prop(self):
        return self._get_reproject_camera_prop()

    def _get_reproject_camera_pose(self):
        return [x for x in self._cfg.policy_obs.reproject.camera_pose]
    
    def get_reproject_camera_pose(self):
        return self._get_reproject_camera_pose()
    
    def get_reproject(
        self, 
        obs_result: Dict[str, np.ndarray], 
        mask_str_to_idx: Dict[str, int], 
        camera_info: Dict[str, Union[CameraProperty, List[float]]],
        interp_mask: bool, 
        target: Literal["double_side", "hanger", "inverse_side"]
    ):
        reproject_cfg = self._cfg.policy_obs.reproject
        if target == "double_side":
            mask_input = np.logical_or(obs_result["mask"] == mask_str_to_idx["garment"], obs_result["mask"] == mask_str_to_idx["garment_inv"]).astype(np.int32)
        elif target == "hanger":
            mask_input = (obs_result["mask"] == mask_str_to_idx["hanger"]).astype(np.int32)
        elif target == "inverse_side":
            mask_input = (obs_result["mask"] == mask_str_to_idx["garment_inv"]).astype(np.int32)
        else:
            raise ValueError(target)
        x1x2y1y2 = np.array(getattr(reproject_cfg.x1x2y1y2, target))

        reproject_result = reproject(
            depth_input=obs_result["depth"], mask_input=mask_input, 
            output_shape=(reproject_cfg.camera_prop.height, reproject_cfg.camera_prop.width),
            intrinsics_matrix_input=camera_property_to_intrinsics_matrix(camera_info["camera_prop"]),
            intrinsics_matrix_output=camera_property_to_intrinsics_matrix(self._get_reproject_camera_prop()),
            camera_pose_input=camera_pose_to_matrix(camera_info["camera_pose"]),
            camera_pose_output=camera_pose_to_matrix(self._get_reproject_camera_pose()),
            interp_mask=interp_mask, x1x2y1y2=x1x2y1y2,
        )
        reproject_info = omegaconf.OmegaConf.to_container(reproject_cfg)
        return reproject_result, reproject_info
    
    def set_gripper_pos(
        self,
        hand_name: Literal["left", "right"],
    ) -> torch.Tensor:
        if hand_name == "left":
            return self._sim_env.robot.get_link_pos(self._left_grasp_link)
        
        elif hand_name == "right":
            return self._sim_env.robot.get_link_pos(self._right_grasp_link)

        else:
            raise NotImplementedError(hand_name)
        
    def _to_full_cfg(self, cfg: Dict[str, torch.Tensor]):
        if cfg is None:
            return None
        full_cfg = {k: v.clone() for k, v in self._sim_env.robot.urdf.cfg.items()}
        for k, v in cfg.items():
            full_cfg[k][...] = v
        return full_cfg

    def set_gripper_target(
        self,
        primitive_name: str, 
        step_name: str, 
        hand_name: Literal["left", "right"], 
        xyz: torch.Tensor, 
        rot: torch.Tensor, 
        init_cfg: Optional[Dict[str, torch.Tensor]] = None, 
        xyz_c: Optional[torch.Tensor] = None, 
        **kwargs, 
    ) -> dict:
        """
        Args:
        - hand_name: str
        - xyz: torch.Tensor, [B, 3]
        - rot: torch.Tensor, [B, 3, 3]
        - init_cfg: Optional, Dict[str, torch.Tensor]
        - xyz_c: Optional, torch.Tensor, [B, 3]
        """
        raise NotImplementedError
    
    def get_rpy(
        self, 
        primitive_name: str, 
        step_name: str, 
        hand_name: Literal["left", "right"], 
        xyz: torch.Tensor, 
        **kwargs, 
    ) -> torch.Tensor:
        """
        Args:
        - xyz: [B, 3]
        
        Return: [B, 3, 3]
        """
        raise NotImplementedError
    
    def set_robot_target_to_init_qpos(self):
        self._sim_env.actor.set_actor_dof_targets(self._init_qpos, self._sim_env.robot_zero_qvel)


class GalbotZeroAgent(BaseAgent):
    def __init__(self, sim_env: SimEnv, agent_cfg: omegaconf.DictConfig) -> None:
        super().__init__(sim_env, agent_cfg)
        raise NotImplementedError

    def get_rpy(
        self, 
        primitive_name: str, 
        step_name: str, 
        hand_name: str, 
        xyz: torch.Tensor, 
        **kwargs,
    ) -> torch.Tensor:
        """
        Args:
        - xyz: [B, 3]
        
        Return: [B, 3, 3]
        """
        def get_rot(primitive_name: str, step_name: str, hand_name: str):
            rpy = getattr(getattr(getattr(self._action_rpy, primitive_name), step_name), hand_name)
            rot = torch.eye(3, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
            rot = torch.tensor(tra.euler_matrix(*(rpy))[:3, :3], dtype=self.dtype, device=self.device)
            return rot

        def raise_error():
            raise NotImplementedError(primitive_name, step_name, hand_name)
        
        try:
            if primitive_name == "fling":
                if step_name == "pick_points":
                    cfg = getattr(getattr(self._action_rpy, primitive_name), step_name)
                    near_rpy = getattr(cfg.near, hand_name)
                    far_rpy = getattr(cfg.far, hand_name) 

                    rot = torch.eye(3, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
                    rot[torch.where(xyz[:, 1] < cfg.y_threshold)[0], ...] = torch.tensor(tra.euler_matrix(*(near_rpy))[:3, :3], dtype=self.dtype, device=self.device)
                    rot[torch.where(torch.logical_not(xyz[:, 1] < cfg.y_threshold))[0], ...] = torch.tensor(tra.euler_matrix(*(far_rpy))[:3, :3], dtype=self.dtype, device=self.device)
                elif step_name in ["lift_up", "fling_backward", "fling_forward1", "fling_forward2", "reset"]:
                    rot = get_rot(primitive_name, step_name, hand_name)
                else:
                    raise_error()
            elif primitive_name == "pick_place":
                if step_name in ["pick_points", "reset"]:
                    rot = get_rot(primitive_name, step_name, hand_name)
                else:
                    raise_error()
            else:
                raise_error()
            
            return rot
        
        except Exception as e:
            print(primitive_name, step_name, hand_name)
            raise e
    
    def set_gripper_target(
        self,
        primitive_name: str, 
        step_name: str, 
        hand_name: Literal["left", "right"], 
        xyz: torch.Tensor, 
        rot: torch.Tensor, 
        init_cfg: Optional[Dict[str, torch.Tensor]] = None, 
        xyz_c: Optional[torch.Tensor] = None, 
        **kwargs, 
    ) -> dict:
        """
        Args:
        - hand_name: str
        - xyz: torch.Tensor, [B, 3]
        - rot: torch.Tensor, [B, 3, 3]
        - init_cfg: Optional, Dict[str, torch.Tensor]
        - xyz_c: Optional, torch.Tensor, [B, 3]
        """
        mat = torch.eye(4, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        mat[:, :3, 3] = xyz
        mat[:, :3, :3] = rot
        
        if hand_name in ["left", "right"]:
            new_cfg, info = self._sim_env.robot.inverse_kinematics(
                target_link=getattr(self._cfg.grasp, hand_name).link,
                target_mat4=mat,
                init_cfg=self._to_full_cfg(init_cfg),
                **(self._ik_solver_cfg),
            )
            logger.info(f"inverse_kinematics:\n{pprint.pformat(info)}")
            target_qpos = {}
            target_qvel = {}
            for joint_name in getattr(self._cfg.grasp, hand_name).joints:
                target_qpos[joint_name] = new_cfg[joint_name]
                target_qvel[joint_name] = self._sim_env.robot_zero_qvel[joint_name]
            self._sim_env.actor.set_actor_dof_targets(target_qpos, target_qvel)

        else:
            raise NotImplementedError(hand_name)
        
        return info


class GalbotOneAgent(BaseAgent):
    def __init__(self, sim_env: SimEnv, agent_cfg: omegaconf.DictConfig) -> None:
        super().__init__(sim_env, agent_cfg)
    
    def get_rpy(
        self, 
        primitive_name: str, 
        step_name: str, 
        hand_name: Literal["left", "right"], 
        xyz: torch.Tensor, 
        **kwargs,
    ) -> Union[torch.Tensor, None]:
        """
        Args:
        - xyz: [B, 3]
        
        Return: [B, 3, 3]
        """
        try:
            rpy = getattr(
                getattr(
                    getattr(
                        self._action_rpy, primitive_name
                    ), step_name
                ), hand_name
            )
        except omegaconf.errors.ConfigAttributeError:
            return None
        rot = torch.eye(3, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        rot = torch.tensor(tra.euler_matrix(*(rpy))[:3, :3], dtype=self.dtype, device=self.device)
        return rot
    
    def set_gripper_target(
        self,
        primitive_name: str, 
        step_name: str, 
        hand_name: Literal["left", "right"], 
        xyz: torch.Tensor, 
        rot: Optional[torch.Tensor], 
        init_cfg: Optional[Dict[str, torch.Tensor]] = None, 
        xyz_c: Optional[torch.Tensor] = None, 
        **kwargs, 
    ) -> dict:
        """
        Args:
        - hand_name: str
        - xyz: torch.Tensor, [B, 3]
        - rot: torch.Tensor, [B, 3, 3]
        - init_cfg: Optional, Dict[str, torch.Tensor]
        - xyz_c: Optional, torch.Tensor, [B, 3]
        """
        # set target matrix
        mat = torch.eye(4, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1)
        mat[:, :3, 3] = xyz
        if rot is not None:
            mat[:, :3, :3] = rot

        # special treatment
        leg_joints = ["leg_joint3", "leg_joint4"]
        if init_cfg is None:
            init_cfg = dict()
        assert isinstance(init_cfg, dict)
        assert isinstance(xyz_c, torch.Tensor)
        urdf = self._sim_env.robot.urdf

        for leg_joint in leg_joints:
            step_cfg = getattr(
                getattr(self._action_rpy, primitive_name, omegaconf.DictConfig({})), 
                step_name, omegaconf.DictConfig({}),
            )

            coeff = getattr(getattr(
                step_cfg, "leg_joint_coeff", omegaconf.DictConfig({})
            ), leg_joint, 0.)
            limit = urdf.joint_map[leg_joint].limit
            if leg_joint == "leg_joint3":
                init_cfg[leg_joint] = (
                    self._init_qpos[leg_joint] + (xyz_c[:, 1] - 0.3).clamp(0., 0.4) * coeff
                ).clamp(limit.lower, limit.upper) # proportional to y value
            elif leg_joint == "leg_joint4":
                init_cfg[leg_joint] = (
                    self._init_qpos[leg_joint] + xyz_c[:, 0].clamp(-0.2, 0.2) * coeff
                ).clamp(limit.lower, limit.upper) # proportional to x value
            else:
                raise ValueError(leg_joint)
            
            # use leg_joint_vel to overwrite:
            if hasattr(step_cfg, "leg_joint_val"):
                val = getattr(step_cfg, "leg_joint_val")
                if hasattr(val, leg_joint):
                    init_cfg[leg_joint] = getattr(val, leg_joint)
        
        # common ik
        if hand_name in ["left", "right"]:
            ee_str = getattr(self._cfg.grasp, hand_name).link
            if rot is not None:
                mask = torch.ones(16, dtype=self.dtype, device=self.device)
            else:
                mask = torch.tensor([0., 0., 0., 1.] * 2 + [1., 0., 0., 1.] + [0., 0., 0., 1.], dtype=self.dtype, device=self.device) # (e'_x)_z = 0
            def err_func(link_transform_map: Dict[str, torch.Tensor]):
                curr_mat4 = link_transform_map[ee_str].view(self.batch_size, 16, 16) # [B, 16, 16]
                err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(self.batch_size, 16) # [B, 16]
                return err_mat * mask
            def loss_func(link_transform_map: Dict[str, torch.Tensor]):
                curr_mat4 = link_transform_map[ee_str]
                err_mat = curr_mat4 - mat # [B, 16]
                return torch.sum(torch.square(err_mat).view(self.batch_size, 16) * mask, dim=1) # [B, ]
            
            new_cfg, info = self._sim_env.robot.inverse_kinematics_optimize(
                err_func=err_func,
                loss_func=loss_func,
                init_cfg=self._to_full_cfg(init_cfg),
                **(self._ik_solver_cfg),
            )
            logger.info(f"inverse_kinematics:\n{pprint.pformat(info)}")
            target_qpos = {}
            target_qvel = {}
            for joint_name in omegaconf.OmegaConf.to_container(
                getattr(self._cfg.grasp, hand_name).joints
            ) + leg_joints:
                target_qpos[joint_name] = new_cfg[joint_name]
                target_qvel[joint_name] = self._sim_env.robot_zero_qvel[joint_name]
            self._sim_env.actor.set_actor_dof_targets(target_qpos, target_qvel)

        else:
            raise NotImplementedError(hand_name)
        
        return info