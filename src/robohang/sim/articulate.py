import taichi as ti

from typing import List, Dict, Union, Tuple, Callable
import copy
import torch
import numpy as np

import trimesh

import omegaconf

import batch_urdf

from .sim_utils import BaseClass
from . import sim_utils
from . import so3
from . import rigid
from ..common import utils

@ti.data_oriented
class Articulate(BaseClass):
    def __init__(self, articulate_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)
        self._name: str = articulate_cfg.name

        self._urdf = batch_urdf.URDF(self._batch_size,
                                     urdf_path=utils.get_path_handler()(articulate_cfg.urdf_path),
                                     dtype=self._dtype,
                                     device=self._device,
                                     mesh_dir=utils.get_path_handler()(articulate_cfg.mesh_dir) if articulate_cfg.mesh_dir is not None else None)
        
        self._collision_map: Dict[str, rigid.RigidMesh] = {}
        self._wrench_map: Dict[str, torch.Tensor] = {
            joint_name: torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device)
            for joint_name in self._urdf.joint_map.keys()
        }
        """dict: str -> tensor [B, 6], force*3 + torque*3 with respect to mass center in local frame (different from rigid body), does not include gravity."""
        self._init_collision_map(articulate_cfg, global_cfg)
        
        self._cfg_pos: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, ...]"""
        self._cfg_vel: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, ...]"""
        for joint_name, joint in self.actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                self._cfg_pos[joint_name] = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
                self._cfg_vel[joint_name] = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
            elif joint.type == "floating":
                self._cfg_pos[joint_name] = sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device)
                self._cfg_vel[joint_name] = sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device)
            else:
                raise NotImplementedError(joint.type)

        self._base_link_pos: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, 7] = 3 translation + 4 quaternion [w, x, y, z]"""
        self._base_link_vel: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, 6] = 3 linear + 3 angular"""
        for l in self._urdf.base_link_map.keys():
            self._base_link_pos[l] = sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device)
            self._base_link_vel[l] = sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device)

        self._link_pos: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, 7] = 3 translation + 4 quaternion [w, x, y, z]"""
        self._link_vel: Dict[str, torch.Tensor] = {}
        """dict: str -> tensor [B, 6] = 3 linear + 3 angular"""
        for l in self._urdf.link_map.keys():
            self._link_pos[l] = sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device)
            self._link_vel[l] = sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device)
        
        self._forward_kinematics(self._cfg_pos, self._cfg_vel)
        for l in self._urdf.base_link_map.keys():
            self._set_forward_base_link_pos_vel(l, self._base_link_pos[l], self._base_link_vel[l])

    def _init_collision_map(self, articulate_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        for link_name in self._urdf.link_map.keys():
            scene = self._urdf.get_link_scene(link_name, True)
            if len(scene.geometry) > 0:
                mesh: trimesh.Trimesh = scene.dump(True)
                rigid_cfg = dict(
                    name=f"{self._name}_{link_name}",
                    sdf_cfg=articulate_cfg.sdf_cfg,
                )
                if link_name not in articulate_cfg.activated_sdf_link:
                    rigid_cfg["sdf_cfg"] = dict(calculate_sdf=False)
                if hasattr(articulate_cfg, "surface_sample"):
                    rigid_cfg["surface_sample"] = articulate_cfg.surface_sample
                self._collision_map[link_name] = rigid.RigidMesh(omegaconf.DictConfig(rigid_cfg), global_cfg, mesh)

    def _to_tensor(self, v) -> torch.Tensor:
        return torch.tensor(v, dtype=self._dtype, device=self._device)

    def _get_joint_pos(self, joint_name: str) -> torch.Tensor:
        joint = self._urdf.joint_map[joint_name]
        if joint.mimic is not None:
            mimic_joint = self._urdf.actuated_joints_map[joint.mimic.joint]
            return self._cfg_pos[mimic_joint.name] * joint.mimic.multiplier + joint.mimic.offset
        else:
            return self._cfg_pos[joint_name].clone()
        
    def get_joint_pos(self, joint_name: str) -> torch.Tensor:
        return self._get_joint_pos(joint_name)
        
    def _get_joint_vel(self, joint_name: str) -> torch.Tensor:
        joint = self._urdf.joint_map[joint_name]
        if joint.mimic is not None:
            mimic_joint = self._urdf.actuated_joints_map[joint.mimic.joint]
            return self._cfg_vel[mimic_joint.name] * joint.mimic.multiplier
        else:
            return self._cfg_vel[joint_name].clone()
        
    def get_joint_vel(self, joint_name: str) -> torch.Tensor:
        return self._get_joint_vel(joint_name)

    def _forward_kinematics_joint(self,
                                  joint: batch_urdf.Joint,
                                  pos: Union[None, torch.Tensor],
                                  vel: Union[None, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Return:
            mat4x4: transform = joint_mat, DOES NOT contain origin
            vel6d: velocity joint_vel, DOES NOT contain origin
        """
        if joint.mimic is not None:
            pos = self._get_joint_pos(joint.name)
            vel = self._get_joint_vel(joint.name)

        if joint.type == "fixed":
            mat4x4 = torch.eye(4, dtype=self._dtype, device=self._device).repeat(self._batch_size, 1, 1)
            vel6d = torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device)
        elif joint.type == "revolute":
            mat4x4 = so3.rotation_matrix(pos, torch.concat([joint.axis] * self._batch_size))
            vel6d = torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device)
            vel6d[:, 3:] = vel[:, None] * joint.axis
        elif joint.type == "prismatic":
            mat4x4 = torch.eye(4, dtype=self._dtype, device=self._device).repeat(self._batch_size, 1, 1)
            mat4x4[:, :3, 3] = pos[:, None] * joint.axis
            vel6d = torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device)
            vel6d[:, :3] = vel[:, None] * joint.axis
        elif joint.type == "floating":
            mat4x4 = so3.pos7d_to_matrix(pos)
            vel6d = vel.clone()
        else:
            raise NotImplementedError(joint.type)

        return mat4x4, vel6d
    
    def _desired_pos_shape(self, joint_name: str):
        j = self._urdf.joint_map[joint_name]
        if j.type in ["revolute", "prismatic"]:
            return (self._batch_size, )
        elif j.type == "floating":
            return (self._batch_size, 7)
        elif j.type == "fixed":
            return ()
        else:
            raise NotImplementedError(f"joint_name:{joint_name}, type:{j.type}")
    
    def _check_pos_shape(self, joint_name: str, pos: torch.Tensor):
        return self._desired_pos_shape(joint_name) == pos.shape
    
    def _desired_vel_shape(self, joint_name: str):
        j = self._urdf.joint_map[joint_name]
        if j.type in ["revolute", "prismatic"]:
            return (self._batch_size, )
        elif j.type == "floating":
            return (self._batch_size, 6)
        elif j.type == "fixed":
            return ()
        else:
            raise NotImplementedError(f"joint_name:{joint_name}, type:{j.type}")
        
    def _check_vel_shape(self, joint_name: str, vel: torch.Tensor):
        return self._desired_vel_shape(joint_name) == vel.shape

    def _set_link_pos_vel(self, link_name: str, pos: torch.Tensor, vel: torch.Tensor, set_rigid=True):
        self._link_pos[link_name][...] = pos
        self._link_vel[link_name][...] = vel
        if set_rigid and link_name in self._collision_map.keys():
            self._collision_map[link_name].set_pos(pos)
            self._collision_map[link_name].set_vel(vel)

    @ti.kernel
    def _fk_compute_pos_vel_kernel(self,
                                   ans_pos7d: ti.types.ndarray(dtype=sim_utils.vec7),
                                   ans_vel6d: ti.types.ndarray(dtype=sim_utils.vec6),
                                   par_vel6d: ti.types.ndarray(dtype=sim_utils.vec6),
                                   dof_vel6d: ti.types.ndarray(dtype=sim_utils.vec6),
                                   par_mat: ti.types.ndarray(dtype=ti.math.mat4),
                                   dof_mat: ti.types.ndarray(dtype=ti.math.mat4),
                                   joint_origin_single_batch: ti.types.ndarray(dtype=ti.math.mat4),
                                   ):
        for batch_idx in range(self._batch_size):
            ans_pos7d[batch_idx] = so3.matrix_to_pos7d_func(
                par_mat[batch_idx] @ joint_origin_single_batch[0] @ dof_mat[batch_idx])
            
            vel = ti.Vector.zero(float, 6)
            dof_rot_world = par_mat[batch_idx][:3, :3] @ joint_origin_single_batch[0][:3, :3]
            
            vel[3:] = dof_rot_world @ dof_vel6d[batch_idx][3:] + par_vel6d[batch_idx][3:]
            vel[:3] = dof_rot_world @ dof_vel6d[batch_idx][:3] + par_vel6d[batch_idx][:3] + \
                par_vel6d[batch_idx][3:].cross(ans_pos7d[batch_idx][:3] - par_mat[batch_idx][:3, 3])
            
            ans_vel6d[batch_idx] = vel

    def _fk_compute_and_set_pos_vel(self, link_name: str, parent_link_name: str, joint_origin_single_batch: torch.Tensor, joint_mat: torch.Tensor, joint_vel: torch.Tensor):
        ans_pos7d = sim_utils.create_zero_7d_pos(self._batch_size, self._dtype, self._device)
        ans_vel6d = sim_utils.create_zero_6d_vel(self._batch_size, self._dtype, self._device)
        self._fk_compute_pos_vel_kernel(ans_pos7d, ans_vel6d,
                                        self._link_vel[parent_link_name], joint_vel,
                                        self._urdf.link_transform_map[parent_link_name], joint_mat,
                                        joint_origin_single_batch)
        self._set_link_pos_vel(link_name, ans_pos7d, ans_vel6d)
    
    @sim_utils.GLOBAL_TIMER.timer
    def _urdf_update_cfg(self, pos):
        self._urdf.update_cfg(pos)

    def _forward_kinematics(self, pos: Dict[str, torch.Tensor], vel: Dict[str, torch.Tensor]):
        """
        Update: self._urdf, self._cfg_pos, self._cfg_vel, self._link_pos, self._link_vel
        """
        self._urdf_update_cfg(pos)
        link_calculated = {link_name: False for link_name in self._urdf.link_map.keys()}

        def _calculate_link_transform(link_name: str) -> None:
            if link_calculated[link_name]:
                return
            elif self._urdf.link_map[link_name].parent_joint is None:
                self._set_link_pos_vel(link_name, self._base_link_pos[link_name], self._base_link_vel[link_name])
            else:
                parent_joint_name = self._urdf.link_map[link_name].parent_joint
                parent_joint = self._urdf.joint_map[parent_joint_name]
                parent_link = parent_joint.parent

                if parent_joint_name not in self.actuated_joints_map.keys():
                    if parent_joint_name in pos.keys():
                        raise RuntimeError(f"joint:{parent_joint_name} in configuration is not actuated. This will not be used.")
                    joint_mat, joint_vel = self._forward_kinematics_joint(parent_joint, None, None)
                elif parent_joint_name not in pos.keys():
                    raise RuntimeError(f"actuated joint:{parent_joint_name} is not in configuration.")
                else:
                    if not self._check_pos_shape(parent_joint_name, pos[parent_joint_name]):
                        raise RuntimeError(f"pos in joint:{parent_joint_name}'s shape {pos[parent_joint_name].shape} is not correct. Desired:{self._desired_pos_shape(parent_joint_name)}")
                    if not self._check_vel_shape(parent_joint_name, vel[parent_joint_name]):
                        raise RuntimeError(f"vel in joint:{parent_joint_name}'s shape {vel[parent_joint_name].shape} is not correct. Desired:{self._desired_vel_shape(parent_joint_name)}")
                    self._cfg_pos[parent_joint_name] = pos[parent_joint_name]
                    self._cfg_vel[parent_joint_name] = vel[parent_joint_name]
                    joint_mat, joint_vel = self._forward_kinematics_joint(parent_joint, pos[parent_joint_name], vel[parent_joint_name])
                
                _calculate_link_transform(parent_link)
                self._fk_compute_and_set_pos_vel(link_name, parent_link, self._urdf.joint_map[parent_joint_name].origin, joint_mat, joint_vel)
                link_calculated[link_name] = True

        for link_name in self._urdf.link_map.keys():
            _calculate_link_transform(link_name)

    @sim_utils.GLOBAL_TIMER.timer
    def forward_kinematics(self, pos: Dict[str, torch.Tensor], vel: Dict[str, torch.Tensor]):
        self._forward_kinematics(pos, vel)

    @sim_utils.GLOBAL_TIMER.timer
    def inverse_kinematics(self, target_link: str, target_mat4: torch.Tensor, **ik_kwargs):
        return self._urdf.inverse_kinematics(target_link, target_mat4, **ik_kwargs)
    
    @sim_utils.GLOBAL_TIMER.timer
    def inverse_kinematics_optimize(self, err_func: Callable, loss_func: Callable, **ik_kwargs):
        return self._urdf.inverse_kinematics_optimize(err_func, loss_func, **ik_kwargs)

    def _set_forward_base_link_pos_vel(self, link_name: str, pos: torch.Tensor, vel: torch.Tensor):
        self._urdf.update_base_link_transformation(link_name, pos)
        self._base_link_pos[link_name][...] = pos
        self._base_link_vel[link_name][...] = vel
        self._forward_kinematics(self._cfg_pos, self._cfg_vel)
    
    def set_forward_base_link_pos_vel(self, link_name: str, pos: torch.Tensor, vel: torch.Tensor):
        """
        Update: self._base_link_pos, self._base_link_vel,

        Then use self._cfg_pos, self._cfg_vel to perform fk.
        """
        self._set_forward_base_link_pos_vel(link_name, pos, vel)

    @property
    def urdf(self) -> batch_urdf.URDF:
        return self._urdf

    @property
    def actuated_joints_map(self) -> Dict[str, batch_urdf.Joint]:
        return self._urdf.actuated_joints_map
    
    @property
    def collision_map(self) -> Dict[str, rigid.RigidMesh]:
        return self._collision_map
    
    @property
    def link_map(self) -> Dict[str, batch_urdf.Link]:
        return self._urdf.link_map
    
    @property
    def joint_map(self) -> Dict[str, batch_urdf.Joint]:
        return self._urdf.joint_map
    
    @property
    def name(self):
        return self._name
    
    @property
    def base_link_map(self) -> Dict[str, batch_urdf.Link]:
        return self._urdf.base_link_map

    def get_cfg_pos(self) -> Dict[str, torch.Tensor]:
        """dict: str -> tensor [B, ...]"""
        return sim_utils.torch_dict_clone(self._cfg_pos)
    
    def get_cfg_vel(self) -> Dict[str, torch.Tensor]:
        """dict: str -> tensor [B, ...]"""
        return sim_utils.torch_dict_clone(self._cfg_vel)
    
    def set_cfg_pos_vel(self, cfg_pos: Dict[str, torch.Tensor]=None, cfg_vel: Dict[str, torch.Tensor]=None):
        """update all configurations which appear in `cfg_pos` and `cfg_vel`, and finally perform fk."""
        if cfg_pos is None:
            cfg_pos = {}
        if cfg_vel is None:
            cfg_vel = {}
        for j in self._urdf.actuated_joints_map.keys():
            if j in cfg_pos.keys():
                self._cfg_pos[j][...] = cfg_pos[j]
            if j in cfg_vel.keys():
                self._cfg_vel[j][...] = cfg_vel[j]
        self._forward_kinematics(self._cfg_pos, self._cfg_vel)

    def _set_drive_force(self, joint_name: str, drive_force: torch.Tensor):
        joint = self._urdf.joint_map[joint_name]
        if joint.type == "floating":
            assert drive_force.shape == (self._batch_size, 6)
            self._wrench_map[joint_name][...] = drive_force
        elif joint.type == "revolute":
            assert drive_force.shape == (self._batch_size, )
            self._wrench_map[joint_name][:, :3] = 0.
            self._wrench_map[joint_name][:, 3:] = drive_force[..., None] * joint.axis
        elif joint.type == "prismatic":
            assert drive_force.shape == (self._batch_size, )
            self._wrench_map[joint_name][:, 3:] = 0.
            self._wrench_map[joint_name][:, :3] = drive_force[..., None] * joint.axis
        else:
            raise NotImplementedError(joint.type)
    
    def get_base_link_pos(self) -> Dict[str, torch.Tensor]:
        """dict: str -> tensor [B, 7] = 3 translation + 4 quaternion [w, x, y, z]"""
        return sim_utils.torch_dict_clone(self._base_link_pos)
    
    def get_base_link_vel(self) -> Dict[str, torch.Tensor]:
        """dict: str -> tensor [B, 6] = 3 linear + 3 angular"""
        return sim_utils.torch_dict_clone(self._base_link_vel)
    
    def get_link_pos(self, link_name: str) -> torch.Tensor:
        """return [B, 7]"""
        return self._link_pos[link_name].clone()
    
    def get_link_vel(self, link_name: str) -> torch.Tensor:
        """return [B, 6]"""
        return self._link_vel[link_name].clone()
    
    def get_actuate_joints_map(self) -> Dict[str, batch_urdf.Joint]:
        return copy.deepcopy(self._urdf.actuated_joints_map)

    def query_sdf(self, position: torch.Tensor, answer_dict: Dict[str, torch.Tensor], query_n: torch.Tensor):
        """
        Args:
            - position: [B, Q, 3]
            - answer_dict: mapping from str to [B, Q, 4]
            - query_n: [B]
        """
        for link_name, answer in answer_dict.items():
            self._collision_map[link_name].query_sdf(position, answer, query_n)
    
    def get_mesh(self, batch_idx: int, use_collision_instead_of_visual: bool = False, **kwargs) -> trimesh.Trimesh:
        return self._urdf.get_scene(batch_idx, use_collision_instead_of_visual).dump(True)
    
    def get_state(self) -> dict:
        return {
            "cfg_pos": {
                k: sim_utils.torch_to_numpy(v) for k, v in self._cfg_pos.items()
            },
            "cfg_vel": {
                k: sim_utils.torch_to_numpy(v) for k, v in self._cfg_vel.items()
            },
            "base_link_pos": {
                k: sim_utils.torch_to_numpy(v) for k, v in self._base_link_pos.items()
            },
            "base_link_vel": {
                k: sim_utils.torch_to_numpy(v) for k, v in self._base_link_vel.items()
            },
        }
    
    def set_state(self, state: dict):
        assert isinstance(state, dict)
        self._forward_kinematics({j: self._to_tensor(v) for j, v in state["cfg_pos"].items()},
                                 {j: self._to_tensor(v) for j, v in state["cfg_vel"].items()})
        for l in self._urdf.base_link_map.keys():
            self._set_forward_base_link_pos_vel(l, 
                                                self._to_tensor(state["base_link_pos"][l]), 
                                                self._to_tensor(state["base_link_vel"][l]))

    def reset(self):
        for joint_name, joint in self.actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                self._cfg_pos[joint_name][...] = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
                self._cfg_vel[joint_name][...] = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
            elif joint.type == "floating":
                self._cfg_pos[joint_name][...] = sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device)
                self._cfg_vel[joint_name][...] = sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device)
            else:
                raise NotImplementedError(joint.type)

        for l in self._urdf.base_link_map.keys():
            self._base_link_pos[l][...] = sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device)
            self._base_link_vel[l][...] = sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device)

        for l in self._urdf.link_map.keys():
            self._set_link_pos_vel(l,
                                   sim_utils.create_zero_7d_pos(self._batch_size, dtype=self._dtype, device=self._device),
                                   sim_utils.create_zero_6d_vel(self._batch_size, dtype=self._dtype, device=self._device))
        
        self._forward_kinematics(self._cfg_pos, self._cfg_vel)
        for l in self._urdf.base_link_map.keys():
            self._set_forward_base_link_pos_vel(l, self._base_link_pos[l], self._base_link_vel[l])
