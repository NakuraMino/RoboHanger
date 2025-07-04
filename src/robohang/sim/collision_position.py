import taichi as ti

from typing import List, Literal, Dict
import copy

import numpy as np
import torch

import omegaconf

from .cloth import Cloth
from .rigid import Rigid
from .articulate import Articulate
from .collision import ClothRigidCollision, ClothCollision
from . import sim_utils
from . import maths


@ti.data_oriented
class ClothPositionCollision(ClothCollision):
    def __init__(self, cloth: Cloth, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._collision_dv: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._cloth._nv))
        """float, [B, V][3]"""

    def _calculate_collision_dv(self, dv: torch.Tensor, dt: float) -> torch.Tensor:
        raise NotImplementedError


@ti.data_oriented
class ClothRigidPositionCollision(ClothRigidCollision, ClothPositionCollision):
    def __init__(self, cloth: Cloth, rigid: Rigid, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        collision_cfg_modified = copy.deepcopy(collision_cfg)
        collision_cfg_modified.max_collision = cloth._nv
        super().__init__(cloth=cloth, rigid=rigid, collision_cfg=collision_cfg_modified, global_cfg=global_cfg, **kwargs)

        self._restitution: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B, ]"""
        self._restitution.fill(collision_cfg_modified.restitution)

        self._max_depenetration_velocity: float = float(collision_cfg_modified.max_depenetration_velocity)
        self._response_time: float = float(collision_cfg_modified.response_time)

    @ti.kernel
    def _calculate_collision_dv_kernel(
        self,
        dv: ti.types.ndarray(dtype=ti.math.vec3),
        dt: float):

        self._generate_sample_on_vert_func(self._query_position, self._query_n)
        self._rigid._query_sdf_func(self._query_position, self._query_answer, self._query_n)

        self._collision_dv.fill(0.0)
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            sdf_and_grad = self._query_answer[batch_idx, vid]

            sdf = sdf_and_grad[0]
            sdf_grad = sdf_and_grad[1:4]

            if sdf < self._balance_distance:

                sample_vel = self._cloth._vel[batch_idx, vid] + dv[batch_idx, vid]

                x0 = self._cloth._pos[batch_idx, vid] - sdf * sdf_grad
                lin_vel = self._rigid._vel[batch_idx][:3]
                ang_vel = self._rigid._vel[batch_idx][3:]
                xyz_pos = self._rigid._pos[batch_idx][:3]
                v0 = lin_vel + ti.math.cross(ang_vel, x0 - xyz_pos)

                rel_vel = sample_vel - v0
                rel_vel_seperation = rel_vel.dot(sdf_grad)
                rel_vel_proj = rel_vel - rel_vel_seperation * sdf_grad
                rel_vel_proj_abs = ti.math.length(rel_vel_proj)
                rel_vel_proj_normed = rel_vel_proj / ti.max(rel_vel_proj_abs, self._dv_eps)

                generalized_seperation_velocity = rel_vel_seperation + (sdf - self._balance_distance) / self._response_time
                if generalized_seperation_velocity < 0.:
                    tmp_dv_1 = ti.min(
                        self._max_depenetration_velocity,
                        -generalized_seperation_velocity * (1. + self._restitution[batch_idx])
                        )
                    tmp_dv_2 = ti.min(tmp_dv_1 * self._mu[batch_idx], rel_vel_proj_abs)
                    tmp_dv = (
                        tmp_dv_1 * sdf_grad -
                        tmp_dv_2 * rel_vel_proj_normed
                    )

                    self._collision_dv[batch_idx, vid] += tmp_dv

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_collision_dv(self, dv: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Args:
            - dv: Tensor [B, V * 3]
            - dt: float
        Return:
            - Tensor [B, V * 3]
        """
        dv = dv.view(self._batch_size, self._cloth._nv, 3)
        self._calculate_collision_dv_kernel(dv, dt)
        return self._collision_dv.to_torch(device=self._device).view(self._batch_size, self._cloth._nv * 3)


@ti.data_oriented
class ClothArticulatePositionCollision(ClothPositionCollision):
    Collision_Iter_Jacobi: Literal[0] = 0
    Collision_Iter_Gauss_Seidel: Literal[1] = 1
    def __init__(self, cloth: Cloth, articulate: Articulate, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._articulate = articulate

        self._collisions: List[ClothRigidPositionCollision] = []
        self._collision_map: Dict[str, ClothRigidPositionCollision] = {}

        for rigid in self._articulate._collision_map.values():
            collision_cfg_rename = omegaconf.DictConfig(collision_cfg)
            collision_cfg_rename.name = collision_cfg.name + f"_{rigid._name}"
            collision = ClothRigidPositionCollision(cloth, rigid, collision_cfg_rename, global_cfg)
            self._collisions.append(collision)
            self._collision_map[rigid._name] = collision

        if collision_cfg.collision_iter_method == "Jacobi":
            self._collision_iter_method = self.Collision_Iter_Jacobi
        elif collision_cfg.collision_iter_method == "GaussSeidel":
            self._collision_iter_method = self.Collision_Iter_Gauss_Seidel
        else:
            raise ValueError(f"Unknown collision iteration method:{collision_cfg.collision_iter_method}")

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_collision_dv(self, dv: torch.Tensor, dt: float) -> torch.Tensor:
        """
        Args:
            - dv: Tensor [B, V * 3]
            - dt: float
        Return:
            - Tensor [B, V * 3]
        """
        # return sum([collision._calculate_collision_dv(dv, dt) for collision in self._collisions])
        acculumalated_dv = torch.zeros_like(dv)
        
        for i in np.random.permutation(len(self._collisions)):
            collision = self._collisions[int(i)]
            if self._collision_iter_method == self.Collision_Iter_Jacobi:
                acculumalated_dv += collision._calculate_collision_dv(
                    dv, dt)
            elif self._collision_iter_method == self.Collision_Iter_Gauss_Seidel:
                acculumalated_dv += collision._calculate_collision_dv(
                    acculumalated_dv + dv, dt)
            else:
                raise ValueError(self._collision_iter_method)
        
        return acculumalated_dv
    
    @property
    def collision_map(self):
        return self._collision_map