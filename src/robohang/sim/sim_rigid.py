import taichi as ti

import torch

import omegaconf

from .sim_utils import BaseClass
from .rigid import Rigid
from . import sim_utils
from . import so3


@ti.func
def rigid_step_vel_func(dt: float,
                        mass: float,
                        inertia: ti.math.mat3,
                        cm_origin: ti.math.mat4,
                        mat_pos: ti.math.mat4,
                        vel: sim_utils.vec6,
                        wrench: sim_utils.vec6):
    """
    Args:
        - dt: float, time step
        - mass: float, rigid mass
        - inertia: mat3x3, inertia w.r.t center of mass
        - cm_origin: mat4x4, center of mass transform
        - mat_pos: mat4x4, in world frame
        - vel: vec6d, in world frame
        - wrench: vec6d, force + torque w.r.t center of mass
    Return:
        - new_vel: vec6d, in world frame
    """
    new_vel = ti.Vector.zero(float, 6)

    mat_cm = mat_pos @ cm_origin
    origin_xyz = mat_cm[:3, 3] - mat_pos[:3, 3]

    # calculate center of mass vel
    vel_cm = so3.fwd_vel_func(vel, origin_xyz)

    # force
    vel_cm[:3] += wrench[:3] * dt / mass

    # torque
    rot_cm = mat_cm[:3, :3]
    inertia_world = rot_cm @ inertia @ rot_cm.transpose()
    drot_pos = so3.angular_velocity_to_drot_dt_func(mat_pos[:3, :3], vel[3:])
    drot_cm = drot_pos @ cm_origin[:3, :3]
    dinertia_world = drot_cm @ inertia @ rot_cm.transpose() + rot_cm @ inertia @ drot_cm.transpose()
    vel_cm[3:] += ti.math.inverse(inertia_world) @ \
        (wrench[3:] - dinertia_world @ vel_cm[3:]) * dt

    # back to rigid vel
    new_vel[:] = so3.fwd_vel_func(vel_cm, -origin_xyz)

    return new_vel
    

@ti.data_oriented
class RigidSim(BaseClass):
    def __init__(self,
                 rigid: Rigid,
                 rigid_sim_cfg: omegaconf.DictConfig,
                 global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)

        self._rigid = rigid
        self._is_dynamic: bool = bool(rigid_sim_cfg.is_dynamic)

        self._total_wrench = sim_utils.GLOBAL_CREATER.VectorField(6, float, shape=(self._batch_size, ))
        """float, [B][6]"""

    @ti.kernel
    def _assemble_total_wrench_kernel(self, 
                                      mass: ti.types.ndarray(dtype=float),
                                      gravity: ti.types.ndarray(dtype=ti.math.vec3),
                                      wrench: ti.types.ndarray(dtype=sim_utils.vec6)):
        for batch_idx in range(self._batch_size):
            self._total_wrench[batch_idx][:] = wrench[batch_idx][:]
            self._total_wrench[batch_idx][:3] += mass[batch_idx] * gravity[batch_idx]

    def _assemble_total_wrench(self, gravity: torch.Tensor):
        self._assemble_total_wrench_kernel(self._rigid._mass, gravity, self._rigid._wrench)

    @ti.kernel
    def _update_velocity_kernel(self,
                                dt: float,
                                mass: ti.types.ndarray(dtype=float),
                                inertia: ti.types.ndarray(dtype=ti.math.mat3),
                                origin: ti.types.ndarray(dtype=ti.math.mat4),
                                pos: ti.template(),
                                vel: ti.template(),):
        for batch_idx in range(self._batch_size):
            vel[batch_idx] = rigid_step_vel_func(dt, mass[batch_idx], inertia[batch_idx], origin[batch_idx], 
                                                 so3.pos7d_to_matrix_func(pos[batch_idx]), vel[batch_idx], self._total_wrench[batch_idx])

    def _step_vel_impl(self, dt: float, gravity: torch.Tensor):
        self._assemble_total_wrench(gravity)
        self._update_velocity_kernel(dt, self._rigid._mass, self._rigid._inertia, self._rigid._inertial_origin, self._rigid._pos, self._rigid._vel)

    def _step_vel_rigid(self, dt: float, gravity: torch.Tensor):
        if self._is_dynamic:
            self._step_vel_impl(dt, gravity)

    @ti.kernel
    def _step_pos_kernel(self, 
                         dt: float,
                         origin: ti.types.ndarray(dtype=ti.math.mat4),
                         ):
        for batch_idx in range(self._batch_size):
            mat_pos = so3.pos7d_to_matrix_func(self._rigid._pos[batch_idx])
            mat_cm = mat_pos @ origin[batch_idx]
            origin_xyz = mat_cm[:3, 3] - mat_pos[:3, 3]

            # calculate center of mass vel
            vel_cm = so3.fwd_vel_func(self._rigid._vel[batch_idx], origin_xyz)

            # update center of mass pos
            mat_cm = so3.update_mat_func(mat_cm, vel_cm, dt)

            # calculate rigid pos
            mat_pos = mat_cm @ so3.inverse_transform_func(origin[batch_idx])

            # update rigid pos and vel
            # here rigid velocity may change, but cm velocity does not change
            self._rigid._pos[batch_idx][:] = so3.matrix_to_pos7d_func(mat_pos)
            self._rigid._vel[batch_idx][:] = so3.fwd_vel_func(vel_cm, -mat_pos[:3, :3] @ origin[batch_idx][:3, 3])

    def _step_pos_rigid(self, dt: float):
        self._step_pos_kernel(dt, self._rigid._inertial_origin)