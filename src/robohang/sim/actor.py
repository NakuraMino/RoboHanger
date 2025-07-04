import taichi as ti

from typing import List, Dict, Union, Literal, Set, Optional
import copy

import torch
import numpy as np

import omegaconf

import batch_urdf
from .sim_utils import BaseClass
from .articulate import Articulate
from .rigid import Rigid
from . import sim_utils
from . import so3

@ti.data_oriented
class Actor(BaseClass):
    Actor_Props_Keys: Set[str] = set(["driveMode", "damping", "stiffness", "integral", "vel_limit"])
    def __init__(self, actor_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)
        self._name: str = actor_cfg.name

    def _step(self, dt: float):
        raise NotImplementedError
    
    @property
    def name(self):
        return self._name
    
    def get_residual(self):
        raise NotImplementedError

    def get_state(self) -> dict:
        return {}
    
    def set_state(self, state: dict):
        assert isinstance(state, dict)
    
    def reset(self):
        return

PropType = Union[str, torch.Tensor, float, None]
"""
driveMode: str
damping: torch.Tensor [B, ]
stiffness: torch.Tensor [B, ]
integral: torch.Tensor [B, ]
vel_limit: float or None
"""

@ti.data_oriented
class ArticulateActor(Actor):
    """
    Actor of an articulate object.

    - For each actuated joints,
        - If `driveMode` == `None`:
            - force = 0
        - If `driveMode` == `PDAcceleration`:
            - force = 0, update cfg_vel directly
        - If `driveMode` == `PIDForce`:
            - set joint's wrench, use articulate simulation to update velocity.
    """
    def __init__(self, articulate: Articulate, actor_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(actor_cfg, global_cfg)
        self._articulate = articulate

        self._actuated_joints_map = self._articulate.actuated_joints_map

        self._actor_props: Dict[str, Dict[str, PropType]] = {}
        self._target_pos: Dict[str, torch.Tensor] = {}
        self._target_vel: Dict[str, torch.Tensor] = {}
        self._accumu_err: Dict[str, torch.Tensor] = {}

        for joint_name, joint in self._actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                self._actor_props[joint_name] = {
                    "driveMode": "None",
                    "damping": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "stiffness": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "integral": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "vel_limit": abs(joint.limit.velocity) if isinstance(joint.limit.velocity, float) else None,
                }
            elif joint.type == "floating":
                self._actor_props[joint_name] = {
                    "driveMode": "None",
                    "damping": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "stiffness": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "integral": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "vel_limit": abs(joint.limit.velocity) if isinstance(joint.limit.velocity, float) else None,
                }
            else:
                raise NotImplementedError(joint.type)
            self._target_pos[joint_name] = self._articulate._cfg_pos[joint_name].clone()
            self._target_vel[joint_name] = self._articulate._cfg_vel[joint_name].clone()
            self._accumu_err[joint_name] = torch.zeros_like(self._target_vel[joint_name])

    def _get_residual(self) -> Dict[Literal["pos", "vel"], Dict[str, torch.Tensor]]:
        residual = {"pos":{}, "vel":{}}
        for joint_name, joint in self._articulate.actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                residual["pos"][joint_name] = (
                    self._target_pos[joint_name] - self._articulate._cfg_pos[joint_name])
                residual["vel"][joint_name] = (
                    self._target_vel[joint_name] - self._articulate._cfg_vel[joint_name])
            elif joint.type == "floating":
                residual["pos"][joint_name] = so3.relative_pos7d_world_frame(
                    self._target_pos[joint_name], self._articulate._cfg_pos[joint_name])
                residual["vel"][joint_name] = (
                    self._target_vel[joint_name] - self._articulate._cfg_vel[joint_name])
            else:
                raise NotImplementedError(joint.type)
        return residual
    
    def get_residual(self):
        return self._get_residual()

    def _step(self, dt: float):
        residual = self._get_residual()
        for joint_name, joint in self._articulate.actuated_joints_map.items():
            actor_props = self._actor_props[joint_name]
            if actor_props["driveMode"] == "None":
                pass
            elif actor_props["driveMode"] == "PDAcceleration":
                self._articulate._cfg_vel[joint_name] += (
                    actor_props["damping"] * residual["vel"][joint_name] +
                    actor_props["stiffness"] * residual["pos"][joint_name]) * dt
            elif actor_props["driveMode"] == "PIDForce":
                self._articulate._set_drive_force(
                    joint_name,
                    actor_props["damping"] * residual["vel"][joint_name] +
                    actor_props["stiffness"] * residual["pos"][joint_name] +
                    actor_props["integral"] * self._accumu_err[joint_name]
                )
                self._accumu_err[joint_name] += dt * residual["pos"][joint_name]
            else:
                raise NotImplementedError(actor_props["driveMode"])
            if isinstance(actor_props["vel_limit"], float):
                self._articulate._cfg_vel[joint_name].clamp_(-actor_props["vel_limit"], +actor_props["vel_limit"])
        self._articulate.forward_kinematics(self._articulate._cfg_pos, self._articulate._cfg_vel)

    def get_actor_dof_properties(self) -> Dict[str, Dict[str, PropType]]:
        return copy.deepcopy(self._actor_props)
    
    def set_actor_dof_properties(self, props: Dict[str, Dict[str, PropType]]):
        for joint_name, prop in props.items():
            if joint_name not in self._actor_props.keys():
                continue

            for k in prop.keys():
                if k == "driveMode":
                    assert isinstance(prop[k], str), props
                    self._actor_props[joint_name][k] = str(prop[k])
                elif k == "vel_limit":
                    assert isinstance(prop[k], (float, type(None))), props
                    self._actor_props[joint_name][k] = prop[k]
                else:
                    assert isinstance(prop[k], torch.Tensor)
                    self._actor_props[joint_name][k][...] = prop[k]

    def set_actor_dof_targets(self, target_pos: Dict[str, torch.Tensor], target_vel: Dict[str, torch.Tensor]):
        for joint_name in target_pos.keys():
            self._target_pos[joint_name][...] = target_pos[joint_name]
        for joint_name in target_vel.keys():
            self._target_vel[joint_name][...] = target_vel[joint_name]

    def clear_accumulate_error(self, joint_name: Optional[str]=None):
        if joint_name is None:
            joint_name_list = self._actuated_joints_map.keys()
        else:
            joint_name_list = [joint_name]
        for j in joint_name_list:
            self._accumu_err[j][...] = 0.

    def get_target(self) -> Dict[Literal["pos", "vel"], Dict[str, torch.Tensor]]:
        target = {"pos":{}, "vel":{}}
        for joint_name, joint in self._articulate.actuated_joints_map.items():
            target["pos"][joint_name] = self._target_pos[joint_name].clone()
            target["vel"][joint_name] = self._target_vel[joint_name].clone()
        return target

    def get_state(self) -> dict:
        state = super().get_state()
        state["props"] = {}
        state["target_pos"] = {}
        state["target_vel"] = {}
        state["accumu_err"] = {}
        for joint_name, joint in self._actuated_joints_map.items():
            state["props"][joint_name] = {}
            for k, v in self._actor_props[joint_name].items():
                if isinstance(v, torch.Tensor):
                    state["props"][joint_name][k] = sim_utils.torch_to_numpy(self._actor_props[joint_name][k])
                elif isinstance(v, (str, float, type(None))):
                    state["props"][joint_name][k] = self._actor_props[joint_name][k]
                else:
                    raise TypeError(type(v))
            state["target_pos"][joint_name] = sim_utils.torch_to_numpy(self._target_pos[joint_name])
            state["target_vel"][joint_name] = sim_utils.torch_to_numpy(self._target_vel[joint_name])
            state["accumu_err"][joint_name] = sim_utils.torch_to_numpy(self._accumu_err[joint_name])
        return state
    
    def set_state(self, state: dict):
        super().set_state(state)
        for joint_name, joint in self._actuated_joints_map.items():
            for k, v in state["props"][joint_name].items():
                assert k in self.Actor_Props_Keys
                if isinstance(v, np.ndarray):
                    self._actor_props[joint_name][k][...] = torch.tensor(state["props"][joint_name][k])
                elif isinstance(v, (str, float, type(None))):
                    self._actor_props[joint_name][k] = state["props"][joint_name][k]
                else:
                    raise TypeError(type(v))
            self._target_pos[joint_name][...] = torch.tensor(state["target_pos"][joint_name])
            self._target_vel[joint_name][...] = torch.tensor(state["target_vel"][joint_name])
            self._accumu_err[joint_name][...] = torch.tensor(state["accumu_err"][joint_name])
            
    def reset(self):
        super().reset()
        for joint_name, joint in self._actuated_joints_map.items():
            if joint.type in ["revolute", "prismatic"]:
                self._actor_props[joint_name] = {
                    "driveMode": "None",
                    "damping": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "stiffness": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "integral": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
                    "vel_limit": abs(joint.limit.velocity) if isinstance(joint.limit.velocity, float) else None,
                }
            elif joint.type == "floating":
                self._actor_props[joint_name] = {
                    "driveMode": "None",
                    "damping": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "stiffness": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "integral": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
                    "vel_limit": abs(joint.limit.velocity) if isinstance(joint.limit.velocity, float) else None,
                }
            else:
                raise NotImplementedError(joint.type)
            self._target_pos[joint_name][...] = self._articulate._cfg_pos[joint_name]
            self._target_vel[joint_name][...] = self._articulate._cfg_vel[joint_name]
            self._accumu_err[joint_name][...] = 0.


@ti.data_oriented
class RigidActor(Actor):
    """
    Actor of an articulate object.
    
    - If `driveMode` == `None`:
        - force = 0
    - If `driveMode` == `PDAcceleration`:
        - force = 0, update rigid velocity directly
    - If `driveMode` == `PIDForce`:
        - set rigid's wrench, use rigid simulation to update velocity. 
        - The `wrench` is exerted on the `center of mass`.
        - The `wrench` is calculated from the pos of the `rigid` (difference of current pos and target pos), not the pos of the center of mass.
          If the frame of `rigid` is different from the frame of `center of mass`, some strange behavior may occur.
    """
    def __init__(self, rigid: Rigid, actor_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(actor_cfg, global_cfg)
        self._rigid = rigid

        self._actor_props: Dict[str, Union[str, torch.Tensor]] = {
            "driveMode": "None",
            "damping": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
            "stiffness": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
            "integral": torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device),
        }
        self._target_pos: torch.Tensor = sim_utils.create_zero_7d_pos(self._batch_size, self._dtype, self._device)
        self._target_vel: torch.Tensor = sim_utils.create_zero_6d_vel(self._batch_size, self._dtype, self._device)
        self._accumu_err: torch.Tensor = torch.zeros_like(self._target_vel)

    def _get_residual(self) -> Dict[Literal["pos", "vel"], torch.Tensor]:
        residual = {"pos":{}, "vel":{}}
        residual["pos"] = so3.relative_pos7d_world_frame(self._target_pos, self._rigid._pos.to_torch(device=self._device))
        residual["vel"] = (self._target_vel - self._rigid._vel.to_torch(device=self._device))
        return residual
    
    def get_residual(self):
        return self._get_residual()

    def _step(self, dt: float):
        residual = self._get_residual()
        if self._actor_props["driveMode"] == "None":
            pass
        elif self._actor_props["driveMode"] == "PDAcceleration":
            self._rigid._vel.from_torch(self._rigid._vel.to_torch(device=self._device) + (
                self._actor_props["damping"] * residual["vel"] +
                self._actor_props["stiffness"] * residual["pos"]) * dt)
        elif self._actor_props["driveMode"] == "PIDForce":
            self._rigid._set_wrench(
                self._actor_props["damping"] * residual["vel"] +
                self._actor_props["stiffness"] * residual["pos"] +
                self._actor_props["integral"] * self._accumu_err
            )
            self._accumu_err += dt * residual["pos"]
        else:
            raise NotImplementedError(self._actor_props["driveMode"])

    def get_actor_properties(self):
        return copy.deepcopy(self._actor_props)
    
    def set_actor_properties(self, props: Dict[str, Union[str, torch.Tensor]]):
        assert "driveMode" in props.keys(), f"{props}"
        for k in props.keys():
            assert k in self.Actor_Props_Keys
            if k == "driveMode":
                self._actor_props[k] = props[k]
            else:
                self._actor_props[k][...] = props[k]

    def set_actor_dof_targets(self, target_pos: torch.Tensor, target_vel: torch.Tensor):
        self._target_pos[...] = target_pos
        self._target_vel[...] = target_vel

    def clear_accumulate_error(self):
        self._accumu_err[...] = 0.

    def get_target(self) -> Dict[Literal["pos", "vel"], torch.Tensor]:
        target = {}
        target["pos"] = self._target_pos.clone()
        target["vel"] = self._target_vel.clone()
        return target

    def get_state(self) -> dict:
        state = super().get_state()
        state["props"] = {}
        for k, v in self._actor_props.items():
            if isinstance(v, torch.Tensor):
                state["props"][k] = sim_utils.torch_to_numpy(self._actor_props[k])
            elif isinstance(v, str):
                state["props"][k] = self._actor_props[k]
            else:
                raise TypeError(type(v))
        state["target_pos"] = sim_utils.torch_to_numpy(self._target_pos)
        state["target_vel"] = sim_utils.torch_to_numpy(self._target_vel)
        state["accumu_err"] = sim_utils.torch_to_numpy(self._accumu_err)
        return state
    
    def set_state(self, state: dict):
        super().set_state(state)
        for k, v in state["props"].items():
            assert k in self.Actor_Props_Keys
            if isinstance(v, np.ndarray):
                self._actor_props[k][...] = torch.tensor(state["props"][k])
            elif isinstance(v, str):
                self._actor_props[k] = state["props"][k]
            else:
                raise TypeError(type(v))
        self._target_pos[...] = torch.tensor(state["target_pos"])
        self._target_vel[...] = torch.tensor(state["target_vel"])
        self._accumu_err[...] = torch.tensor(state["accumu_err"])
            
    def reset(self):
        super().reset()
        self._actor_props = {
            "driveMode": "None",
            "damping": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
            "stiffness": torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device),
        }
        self._target_pos[...] = sim_utils.create_zero_7d_pos(self._batch_size, self._dtype, self._device)
        self._target_vel[...] = sim_utils.create_zero_6d_vel(self._batch_size, self._dtype, self._device)
        self._accumu_err[...] = torch.zeros_like(self._target_vel)
