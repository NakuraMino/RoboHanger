import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import Union, List, Literal, Optional, Dict
import torch

import omegaconf

from .cloth import Cloth
from .rigid import Rigid
from .articulate import Articulate
from .sparse import SparseMatrix
from .collision import ClothCollision, ClothRigidCollision, FaceSampleInfo
from . import maths
from . import spatial
from . import sim_utils
from . import so3


@ti.func
def polynomial_barrier_function_func(x: float, d: float, k: float, c: float) -> ti.math.vec3:
    """
    Args:
        - x, sdf: float
        - d, barrier_width: float
        - k, barrier_power: float
        - c, balance_distance: float
    
    Function:
        - `B(x) = [max((c - x) / d, 0.0)] ^ k`
        - ret_val[1] < 0
    """
    ret_val = ti.Vector.zero(dt=float, n=3)

    if x < c:
        ret_val[0] = ((c - x) / d) ** k
        ret_val[1] = - k * ((c - x) / d) ** (k - 1) / d
        ret_val[2] = k * (k - 1) * ((c - x) / d) ** (k - 2) / (d ** 2)

    return ret_val


@ti.func
def smoothed_friction_func(rel_vel_abs: float, eps_vel: float) -> ti.math.vec2:
    """
    Args:
        - rel_vel_abs: u, absolute value of relative velocity
        - eps_velocity: e, tolerance velocity

    Function:
        - f(u) = 
            - `-(u/e)**2 + 2u/e` if `0<=u<e`
            - `1` if `u>=e`
        - f'(u) >= 0

    Return: f(u) and f'(u)
    """
    ret_val = ti.Vector([1., 0.], dt=float)
    assert rel_vel_abs >= 0. and eps_vel > 0., f"{rel_vel_abs}, {eps_vel}"
    if 0 <= rel_vel_abs and rel_vel_abs < eps_vel:
        ret_val[0] = - (rel_vel_abs / eps_vel) ** 2 + 2 * (rel_vel_abs / eps_vel)
        ret_val[1] = - 2 * rel_vel_abs / (eps_vel ** 2) + 2 / eps_vel
    return ret_val


@ti.data_oriented
class ClothForceCollision(ClothCollision):
    def __init__(self, cloth: Cloth, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._barrier_width: float = float(collision_cfg.barrier_width)
        self._barrier_power: float = float(collision_cfg.get("barrier_power", 3.))
        self._barrier_strength: float = float(collision_cfg.barrier_strength)

        self._cloth_penalty_force: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._cloth._nv))
        """float, [B, V][3]"""
        self._cloth_penalty_df_dt: Optional[ti.MatrixField]
        """float, [B, V][3]"""

        self._cloth_hessian_sparse: Optional[SparseMatrix]
        """Intermediate variables could be `None`"""
    
    def _calculate_penalty_force(self, dt: float, **kwargs):
        raise NotImplementedError


@ti.data_oriented
class ClothSelfForceCollision(ClothForceCollision):
    def __init__(self, cloth: Cloth, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._max_vert_face_collision_pair: int = int(collision_cfg.max_vert_face_collision_pair)
        self._max_edge_edge_collision_pair: int = int(collision_cfg.max_edge_edge_collision_pair)

        self._vert_face_collision_cnt: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))
        """int, [B, ]"""
        self._edge_edge_collision_cnt: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))
        """int, [B, ]"""
        self._cloth_penalty_df_dt = None

        self._mu: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B, ]"""
        self._mu.fill(collision_cfg.mu)
        self._friction_relative_velocity_scale = float(collision_cfg.friction_relative_velocity_scale)

        self._dv_eps: float = float(collision_cfg.dv_eps)
        assert self._dv_eps > 0.

        self._cloth_hessian_sparse = SparseMatrix(
            self._batch_size,
            self._cloth._nv * 3,
            self._cloth._nv * 3,
            (self._max_vert_face_collision_pair + self._max_edge_edge_collision_pair) * (12 ** 2),
            False,
        )

    @ti.func
    def _vert_face_dd_dX_func(
        self,
        x: ti.math.vec3,
        a: ti.math.vec3,
        b: ti.math.vec3,
        c: ti.math.vec3,
        u: float, 
        v: float, 
        w: float
    ) -> ti.types.matrix(4, 3, float):
        ret_val = ti.Matrix.zero(float, 4, 3)
        vec = x - (a * u + b * v + c * w)
        vec_normed = maths.safe_normalized_func(vec, self._dx_eps)
        ret_val[0, :] = vec_normed
        ret_val[1, :] = - u * vec_normed
        ret_val[2, :] = - v * vec_normed
        ret_val[3, :] = - w * vec_normed
        return ret_val
    
    @ti.func
    def _edge_edge_dd_dX_func(
        self,
        a: ti.math.vec3,
        b: ti.math.vec3,
        c: ti.math.vec3,
        d: ti.math.vec3,
        u: float, 
        v: float
    ) -> ti.types.matrix(4, 3, float):
        ret_val = ti.Matrix.zero(float, 4, 3)
        vec = (a * u + b * (1. - u)) - (c * v + d * (1. - v))
        vec_normed = maths.safe_normalized_func(vec, self._dx_eps)
        ret_val[0, :] = vec_normed * u
        ret_val[1, :] = vec_normed * (1. - u)
        ret_val[2, :] = - vec_normed * v
        ret_val[3, :] = - vec_normed * (1. - v)
        return ret_val
    
    @ti.func
    def _expanded_face_bounding_box(self, batch_idx: int, fid: int) -> spatial.BoundingBox:
        """expand by `self._balance_distance / 2`"""
        vids = self._cloth._f2v[fid]
        x0 = self._cloth._pos[batch_idx, vids[0]]
        x1 = self._cloth._pos[batch_idx, vids[1]]
        x2 = self._cloth._pos[batch_idx, vids[2]]
        bb = spatial.triangle_bounding_box_func(x0, x1, x2)
        bb.bounds[0, :] -= self._balance_distance / 2
        bb.bounds[1, :] += self._balance_distance / 2
        return bb
    
    @ti.func
    def _expanded_vert_bounding_box(self, batch_idx: int, vid: int) -> spatial.BoundingBox:
        """expand by `self._balance_distance / 2`"""
        x = self._cloth._pos[batch_idx, vid]
        bb = spatial.point_bounding_box_func(x)
        bb.bounds[0, :] -= self._balance_distance / 2
        bb.bounds[1, :] += self._balance_distance / 2
        return bb
    
    @ti.func
    def _expanded_edge_bounding_box(self, batch_idx: int, eid: int) -> spatial.BoundingBox:
        """expand by `self._balance_distance / 2`"""
        vids = self._cloth._e2v[eid]
        x0 = self._cloth._pos[batch_idx, vids[0]]
        x1 = self._cloth._pos[batch_idx, vids[1]]
        bb = spatial.line_bounding_box_func(x0, x1)
        bb.bounds[0, :] -= self._balance_distance / 2
        bb.bounds[1, :] += self._balance_distance / 2
        return bb

    @ti.kernel
    def _calculate_penalty_force_init_kernel(self):
        self._fatal_flag.fill(False)
        self._vert_face_collision_cnt.fill(0)
        self._edge_edge_collision_cnt.fill(0)
        self._cloth_penalty_force.fill(0.0)
        self._cloth_hessian_sparse.set_zero_func()

    @ti.func
    def _add_single_vert_face_pair_collision_func(self, batch_idx: int, dt: float, vid: int, fid: int, vert_face_mask):
        if ((vid != self._cloth._f2v[fid]).all() and 
            (not ti.atomic_or(vert_face_mask[batch_idx, vid, fid], 1))):
            # vert_face_mask[batch_idx, vid, fid] = 1
            vids = ti.Vector([
                vid,
                self._cloth._f2v[fid][0],
                self._cloth._f2v[fid][1],
                self._cloth._f2v[fid][2],
            ], dt=int)

            x = self._cloth._pos[batch_idx, vids[0]]
            a = self._cloth._pos[batch_idx, vids[1]]
            b = self._cloth._pos[batch_idx, vids[2]]
            c = self._cloth._pos[batch_idx, vids[3]]
            l, u, v, w = maths.get_distance_to_triangle_func(x, a, b, c, self._dx_eps)

            balance_distance = self._balance_distance

            if l < balance_distance:
                old_cnt = ti.atomic_add(self._vert_face_collision_cnt[batch_idx], +1)
                if old_cnt < self._max_vert_face_collision_pair:
                    collision_volume = self._cloth._get_vertex_volume_func(batch_idx, vid)
                    e0, e1, e2 = self._barrier_strength * \
                        collision_volume * \
                        polynomial_barrier_function_func(
                            l,
                            self._barrier_width,
                            self._barrier_power,
                            balance_distance,
                        )
                    dd_dX = self._vert_face_dd_dX_func(x, a, b, c, u, v, w) # [4, 3]

                    for i in range(4):
                        self._cloth_penalty_force[batch_idx, vids[i]] += -e1 * dd_dX[i, :]

                    for i, j, m, n in ti.ndrange(4, 4, 3, 3):
                        self._cloth_hessian_sparse.add_value_func(
                            batch_idx,
                            vids[i] * 3 + m,
                            vids[j] * 3 + n,
                            e2 * dd_dX[i, m] * dd_dX[j, n]
                        )
                    
                    vel_x = self._cloth._vel[batch_idx, vids[0]]
                    vel_p = self._cloth._vel[batch_idx, vids[1]] * u + self._cloth._vel[batch_idx, vids[2]] * v + self._cloth._vel[batch_idx, vids[3]] * w
                    norm_xp = maths.safe_normalized_func(x - (a * u + b * v + c * w), self._dv_eps)
                    vel_xp = vel_x - vel_p
                    vel_xp_proj = vel_xp - vel_xp.dot(norm_xp) * norm_xp

                    mass = collision_volume * self._cloth._rho[batch_idx]
                    friction_force = ti.min(
                        -e1 * self._mu[batch_idx],
                        self._friction_relative_velocity_scale * ti.math.length(vel_xp_proj) * mass / dt,
                    )
                    
                    self._cloth_penalty_force[batch_idx, vids[0]] -= friction_force * maths.safe_normalized_func(vel_xp_proj, self._dv_eps)
                    self._cloth_penalty_force[batch_idx, vids[1]] += friction_force * maths.safe_normalized_func(vel_xp_proj, self._dv_eps) / 3
                    self._cloth_penalty_force[batch_idx, vids[2]] += friction_force * maths.safe_normalized_func(vel_xp_proj, self._dv_eps) / 3
                    self._cloth_penalty_force[batch_idx, vids[3]] += friction_force * maths.safe_normalized_func(vel_xp_proj, self._dv_eps) / 3

                else:
                    ti.atomic_add(self._vert_face_collision_cnt[batch_idx], -1)
                    old_fatal_flag = ti.atomic_or(self._fatal_flag[batch_idx], True)
                    if not old_fatal_flag:
                        print(f"[ERROR] batch_idx {batch_idx} self vert-face collision query number reaches maximum {self._max_vert_face_collision_pair}.")

    @ti.kernel
    def _calculate_penalty_force_vert_face_kernel(
        self, 
        dt: float,  
        spatial_partition: ti.template(), 
        good_batch_indices: ti.types.ndarray(dtype=int), 
        vert_face_mask: ti.template(),
        max_cnt_1: ti.types.ndarray(dtype=int), 
        all_cnt_1: ti.types.ndarray(dtype=int),
        max_cnt_2: ti.types.ndarray(dtype=int), 
        all_cnt_2: ti.types.ndarray(dtype=int),
    ):
        # spatial data structure
        for batch_idx_, fid in ti.ndrange(good_batch_indices.shape[0], self._cloth._nf):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._expanded_face_bounding_box(batch_idx, fid)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_1[batch_idx], cnt)
            all_cnt_1[batch_idx] += cnt
            spatial_partition.add_bounding_box_func(batch_idx, bb, fid)

        # vert-face collision
        for batch_idx_, vid in ti.ndrange(good_batch_indices.shape[0], self._cloth._nv):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._expanded_vert_bounding_box(batch_idx, vid)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_2[batch_idx], cnt)
            all_cnt_2[batch_idx] += cnt
            if cnt <= spatial_partition._max_bb_occupy_num:
                ti.loop_config(serialize=True)
                for i, j, k in ti.ndrange(
                    (ijk_lower[0], ijk_upper[0] + 1),
                    (ijk_lower[1], ijk_upper[1] + 1),
                    (ijk_lower[2], ijk_upper[2] + 1),):
                    ti.loop_config(serialize=True)
                    for l in range(spatial_partition.get_cell_length_func(batch_idx, i, j, k)):
                        fid = spatial_partition.get_cell_item_func(batch_idx, i, j, k, l)
                        self._add_single_vert_face_pair_collision_func(batch_idx, dt, vid, fid, vert_face_mask)
            else:
                old_fatal_flag = ti.atomic_or(spatial_partition._fatal_flag[batch_idx], True)
                if not old_fatal_flag:
                    print(f"[ERROR] batch_idx {batch_idx} self vert-face collision loop size: {cnt} is too large")
    
    @ti.func
    def _add_single_edge_edge_pair_collision_func(self, batch_idx: int, dt: float, e1id: int, e2id: int, edge_edge_mask):
        if ((self._cloth._e2v[e1id][0] != self._cloth._e2v[e2id]).all() and
            (self._cloth._e2v[e1id][1] != self._cloth._e2v[e2id]).all() and
            (not ti.atomic_or(edge_edge_mask[batch_idx, e1id, e2id], 1)) and (e1id < e2id)):
            # edge_edge_mask[batch_idx, e1id, e2id] = 1
            vids = ti.Vector([
                self._cloth._e2v[e1id][0],
                self._cloth._e2v[e1id][1],
                self._cloth._e2v[e2id][0],
                self._cloth._e2v[e2id][1],
            ], dt=int)
            
            a = self._cloth._pos[batch_idx, vids[0]]
            b = self._cloth._pos[batch_idx, vids[1]]
            c = self._cloth._pos[batch_idx, vids[2]]
            d = self._cloth._pos[batch_idx, vids[3]]
            l, u, v = maths.get_distance_edge_edge_func(a, b, c, d, self._dx_eps)

            balance_distance = self._balance_distance

            if l < balance_distance:
                old_cnt = ti.atomic_add(self._edge_edge_collision_cnt[batch_idx], +1)
                if old_cnt < self._max_edge_edge_collision_pair:
                    collision_volume = (
                        self._cloth._get_edge_volume_func(batch_idx, e1id) + 
                        self._cloth._get_edge_volume_func(batch_idx, e2id)
                    ) / 2
                    e0, e1, e2 = self._barrier_strength * \
                        collision_volume * \
                        polynomial_barrier_function_func(
                            l,
                            self._barrier_width,
                            self._barrier_power,
                            balance_distance,
                        )
                    dd_dX = self._edge_edge_dd_dX_func(a, b, c, d, u, v) # [4, 3]

                    for i in range(4):
                        self._cloth_penalty_force[batch_idx, vids[i]] += -e1 * dd_dX[i, :]

                    for i, j, m, n in ti.ndrange(4, 4, 3, 3):
                        self._cloth_hessian_sparse.add_value_func(
                            batch_idx,
                            vids[i] * 3 + m,
                            vids[j] * 3 + n,
                            e2 * dd_dX[i, m] * dd_dX[j, n]
                        )

                    vel_u = self._cloth._vel[batch_idx, vids[0]] * u + self._cloth._vel[batch_idx, vids[1]] * (1. - u)
                    vel_v = self._cloth._vel[batch_idx, vids[2]] * v + self._cloth._vel[batch_idx, vids[3]] * (1. - v)
                    norm_uv = maths.safe_normalized_func((u * a + (1. - u) * b) - (v * c + (1. - v) * d), self._dv_eps)
                    vel_uv = vel_u - vel_v
                    vel_uv_proj = vel_uv - vel_uv.dot(norm_uv) * norm_uv

                    mass = collision_volume * self._cloth._rho[batch_idx]
                    friction_force = ti.min(
                        -e1 * self._mu[batch_idx],
                        self._friction_relative_velocity_scale * ti.math.length(vel_uv_proj) * mass / dt,
                    )
                    
                    self._cloth_penalty_force[batch_idx, vids[0]] -= friction_force * maths.safe_normalized_func(vel_uv_proj, self._dv_eps) / 2.
                    self._cloth_penalty_force[batch_idx, vids[1]] -= friction_force * maths.safe_normalized_func(vel_uv_proj, self._dv_eps) / 2.
                    self._cloth_penalty_force[batch_idx, vids[2]] += friction_force * maths.safe_normalized_func(vel_uv_proj, self._dv_eps) / 2.
                    self._cloth_penalty_force[batch_idx, vids[3]] += friction_force * maths.safe_normalized_func(vel_uv_proj, self._dv_eps) / 2.

                else:
                    ti.atomic_add(self._edge_edge_collision_cnt[batch_idx], -1)
                    old_fatal_flag = ti.atomic_or(self._fatal_flag[batch_idx], True)
                    if not old_fatal_flag:
                        print(f"[ERROR] batch_idx {batch_idx} self edge-edge collision query number reaches maximum {self._max_edge_edge_collision_pair}.")

    @ti.kernel
    def _calculate_penalty_force_edge_edge_kernel(
        self, 
        dt: float,
        spatial_partition: ti.template(), 
        good_batch_indices: ti.types.ndarray(dtype=int), 
        edge_edge_mask: ti.template(),
        max_cnt_1: ti.types.ndarray(dtype=int), 
        all_cnt_1: ti.types.ndarray(dtype=int),
        max_cnt_2: ti.types.ndarray(dtype=int), 
        all_cnt_2: ti.types.ndarray(dtype=int),
    ):
        # spatial data structure
        for batch_idx_, e1id in ti.ndrange(good_batch_indices.shape[0], self._cloth._ne):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._expanded_edge_bounding_box(batch_idx, e1id)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_1[batch_idx], cnt)
            all_cnt_1[batch_idx] += cnt
            spatial_partition.add_bounding_box_func(batch_idx, bb, e1id)

        # edge-edge collision
        for batch_idx_, e2id in ti.ndrange(good_batch_indices.shape[0], self._cloth._ne):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._expanded_edge_bounding_box(batch_idx, e2id)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_2[batch_idx], cnt)
            all_cnt_2[batch_idx] += cnt
            if cnt <= spatial_partition._max_bb_occupy_num:
                ti.loop_config(serialize=True)
                for i, j, k in ti.ndrange(
                    (ijk_lower[0], ijk_upper[0] + 1),
                    (ijk_lower[1], ijk_upper[1] + 1),
                    (ijk_lower[2], ijk_upper[2] + 1),):
                    ti.loop_config(serialize=True)
                    for l in range(spatial_partition.get_cell_length_func(batch_idx, i, j, k)):
                        e1id = spatial_partition.get_cell_item_func(batch_idx, i, j, k, l)
                        self._add_single_edge_edge_pair_collision_func(batch_idx, dt, e1id, e2id, edge_edge_mask)
            else:
                old_fatal_flag = ti.atomic_or(spatial_partition._fatal_flag[batch_idx], True)
                if not old_fatal_flag:
                    print(f"[ERROR] batch_idx {batch_idx} self edge-edge collision loop size: {cnt} is too large")

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_penalty_force(
        self, 
        dt: float, 
        spatial_partition: spatial.SpatialPartition, 
        global_fatal_flag: torch.Tensor,
        vert_face_mask: ti.ScalarField,
        vert_face_mask_pointer: ti.SNode,
        edge_edge_mask: ti.ScalarField,
        edge_edge_mask_pointer: ti.SNode,
        **kwargs,
    ):
        vf_max_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_all_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_max_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_all_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_max_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_all_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_max_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_all_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)

        self._calculate_penalty_force_init_kernel()

        vert_face_mask_pointer.deactivate_all()
        spatial_partition.deactivate_all_kernel()
        self._calculate_penalty_force_vert_face_kernel(dt, spatial_partition, torch.where(global_fatal_flag == 0)[0].to(dtype=self._dtype_int), vert_face_mask, vf_max_cnt_1, vf_all_cnt_1, vf_max_cnt_2, vf_all_cnt_2)
        global_fatal_flag |= spatial_partition.get_fatal_flag()

        edge_edge_mask_pointer.deactivate_all()
        spatial_partition.deactivate_all_kernel()
        self._calculate_penalty_force_edge_edge_kernel(dt, spatial_partition, torch.where(global_fatal_flag == 0)[0].to(dtype=self._dtype_int), edge_edge_mask, ee_max_cnt_1, ee_all_cnt_1, ee_max_cnt_2, ee_all_cnt_2)
        global_fatal_flag |= spatial_partition.get_fatal_flag()

        global_fatal_flag |= self._fatal_flag.to_torch(device=self._device)

        if self._log_verbose >= 1:
            logger.debug(f"{self._name} self vert-face {self._vert_face_collision_cnt}")
            logger.debug(f"{self._name} self edge-edge {self._edge_edge_collision_cnt}")

        if self._log_verbose >= 2:
            logger.debug(f"{self._name} self vf_max_cnt_1:{vf_max_cnt_1}")
            logger.debug(f"{self._name} self vf_all_cnt_1:{vf_all_cnt_1}")
            logger.debug(f"{self._name} self vf_max_cnt_2:{vf_max_cnt_2}")
            logger.debug(f"{self._name} self vf_all_cnt_2:{vf_all_cnt_2}")
            logger.debug(f"{self._name} self ee_max_cnt_1:{ee_max_cnt_1}")
            logger.debug(f"{self._name} self ee_all_cnt_1:{ee_all_cnt_1}")
            logger.debug(f"{self._name} self ee_max_cnt_2:{ee_max_cnt_2}")
            logger.debug(f"{self._name} self ee_all_cnt_2:{ee_all_cnt_2}")


@ti.data_oriented
class ClothRigidForceCollision(ClothRigidCollision, ClothForceCollision):
    IPC_Friction: Literal[0] = 0
    Coulomb_Friction: Literal[1] = 1

    def __init__(self, cloth: Cloth, rigid: Rigid, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, rigid=rigid, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        if collision_cfg.friction_type == "IPC":
            self._friction_type = self.IPC_Friction
            self._eps_vel: float = float(collision_cfg.eps_vel)
            assert self._eps_vel > 0.0
        elif collision_cfg.friction_type == "Coulomb":
            self._friction_type = self.Coulomb_Friction
        else:
            raise ValueError(f"Unknown friction type:{collision_cfg.friction_type}")
        
        self._sample_dx: float = float(collision_cfg.sample_dx)
        assert self._sample_dx > 0.
        self._face_sample_cnt: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, self._cloth._nf))
        """int, [B, F]"""
        self._face_sample_info: ti.StructField = sim_utils.GLOBAL_CREATER.StructField(FaceSampleInfo, shape=(self._batch_size, self._max_collision))
        """[int, (float, float)], [B, Q]"""

        self._cloth_penalty_df_dt: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._cloth._nv))
        """float, [B, V][3]"""
        self._cloth_hessian_matrix: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(3, 3, float, shape=(self._batch_size, self._cloth._nf, 3, 3))
        """float, [B, F, 3, 3][3, 3]"""

        self._penetration: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))
        """int, [B]"""

        self._friction_relative_velocity_scale = float(collision_cfg.friction_relative_velocity_scale)

        self._store_sparse_matrix: bool = collision_cfg.get("store_sparse_matrix", True)
        if self._store_sparse_matrix:
            self._cloth_hessian_sparse = SparseMatrix(
                self._batch_size,
                self._cloth._nv * 3,
                self._cloth._nv * 3,
                self._cloth._nf * (9 ** 2),
                False,
            )

    @ti.func
    def _calculate_penalty_force_func(self, query_answer, query_n, query_position, rigid_pos, rigid_vel, dt):
        self._cloth_penalty_force.fill(0.0)
        self._cloth_penalty_df_dt.fill(0.0)
        self._cloth_hessian_matrix.fill(0.0)
        query_n_max = maths.vec_max_func(query_n, self._batch_size)
        for batch_idx, sample_idx in ti.ndrange(self._batch_size, query_n_max):
            if sample_idx < query_n[batch_idx]:
                sdf_and_grad = query_answer[batch_idx, sample_idx]

                sdf = sdf_and_grad[0]
                sdf_grad = sdf_and_grad[1:4]

                if sdf < self._balance_distance:
                    sample_info = self._face_sample_info[batch_idx, sample_idx]
                    fid = sample_info.face_id

                    vids = self._cloth._f2v[fid]
                    bc = ti.Vector([sample_info.bary_coor[0], 
                                    sample_info.bary_coor[1],
                                    1.0 - sample_info.bary_coor[0] - sample_info.bary_coor[1]], float)
                    sample_volume = (
                        self._cloth._get_face_rest_volume_func(batch_idx, fid) /
                        self._face_sample_cnt[batch_idx, fid]
                    )

                    e0, e1, e2 = self._barrier_strength * sample_volume * \
                        polynomial_barrier_function_func(
                            sdf,
                            self._barrier_width,
                            self._barrier_power,
                            self._balance_distance,
                        )
                    
                    force = (-e1) * sdf_grad
                    penalty_hessian = e2 * sdf_grad.outer_product(sdf_grad)
                    for i in ti.static(range(3)):
                        self._cloth_penalty_force[batch_idx, vids[i]] += force * bc[i] # Fc

                    x0 = query_position[batch_idx, sample_idx] - sdf * sdf_grad
                    lin_vel = rigid_vel[batch_idx][:3]
                    ang_vel = rigid_vel[batch_idx][3:]
                    xyz_pos = rigid_pos[batch_idx][:3]
                    v0 = lin_vel + ti.math.cross(ang_vel, x0 - xyz_pos)
                    df_dt = e2 * sdf_grad.dot(v0) * sdf_grad
                    for i in ti.static(range(3)):
                        self._cloth_penalty_df_dt[batch_idx, vids[i]] += df_dt * bc[i] # dFc / dt

                    for i, j in ti.static(ti.ndrange(3, 3)):
                        self._cloth_hessian_matrix[batch_idx, fid, i, j] += penalty_hessian * bc[i] * bc[j] # -dFc / dx

                    # estimate friction force
                    sample_vel = (bc[0] * self._cloth._vel[batch_idx, vids[0]] +
                                  bc[1] * self._cloth._vel[batch_idx, vids[1]] +
                                  bc[2] * self._cloth._vel[batch_idx, vids[2]])
                    mass = sample_volume * self._cloth._rho[batch_idx]
                    dv_guess = (
                        ti.Matrix.identity(float, 3) * mass +
                        dt * dt * penalty_hessian
                    ).inverse() @ (
                        force + df_dt * dt - penalty_hessian @ sample_vel * dt
                    ) * dt

                    rel_vel = sample_vel + dv_guess - v0
                    rel_vel_proj = rel_vel - rel_vel.dot(sdf_grad) * sdf_grad
                    rel_vel_proj_abs = ti.math.length(rel_vel_proj)
                    rel_vel_proj_normed = rel_vel_proj / ti.max(rel_vel_proj_abs, self._dv_eps)

                    if ti.static(self._friction_type == self.IPC_Friction):
                        f1, f2 = smoothed_friction_func(rel_vel_proj_abs, self._eps_vel)
                        friction_force = self._mu[batch_idx] * ti.math.length(dv_guess * mass / dt)

                        for i in ti.static(range(3)):
                            self._cloth_penalty_force[batch_idx, vids[i]] += (
                                -friction_force * f1 * rel_vel_proj_normed * bc[i]
                            ) # Ff

                        rel_vel_proj_normed_outer_product = rel_vel_proj_normed.outer_product(rel_vel_proj_normed)
                        ddu_dx = (ti.Matrix.identity(float, 3) - 
                                rel_vel_proj_normed_outer_product) / ti.max(rel_vel_proj_abs, self._dv_eps)
                        for i, j in ti.static(ti.ndrange(3, 3)):
                            hess = (
                                (friction_force * f2 / dt * 
                                rel_vel_proj_normed_outer_product *
                                bc[i] * bc[j] +
                                friction_force * f1 / dt *
                                ddu_dx *
                                bc[i] * bc[j]) # -dFf / dx
                            )
                            self._cloth_hessian_matrix[batch_idx, fid, i, j] += hess
                    
                    elif ti.static(self._friction_type == self.Coulomb_Friction):
                        friction_force = ti.min(
                            self._mu[batch_idx] * ti.math.length(dv_guess * mass / dt),
                            self._friction_relative_velocity_scale * rel_vel_proj_abs * mass / dt,
                        )

                        for i in ti.static(range(3)):
                            self._cloth_penalty_force[batch_idx, vids[i]] += (
                                -friction_force * rel_vel_proj_normed * bc[i]
                            ) # Ff

        if ti.static(self._store_sparse_matrix):
            self._cloth_hessian_sparse.set_zero_func()
            for batch_idx, fid, i, j, m, n in ti.ndrange(self._batch_size, self._cloth._nf, 3, 3, 3, 3):
                vids = self._cloth._f2v[fid]
                self._cloth_hessian_sparse.add_value_func(
                    batch_idx,
                    vids[i] * 3 + m,
                    vids[j] * 3 + n,
                    self._cloth_hessian_matrix[batch_idx, fid, i, j][m, n]
                )

    @ti.kernel
    def _calculate_penalty_force_kernel(self, dt: float):
        self._generate_sample_on_vert_func(self._query_position, self._query_n)
        self._rigid._query_sdf_func(self._query_position, self._query_answer, self._query_n)
        self._generate_sample_on_face_func(self._query_position, self._query_n, self._query_answer, 
                                           self._face_sample_info, self._face_sample_cnt, self._sample_dx, self._balance_distance)
        self._rigid._query_sdf_func(self._query_position, self._query_answer, self._query_n)
        self._calculate_penalty_force_func(self._query_answer, self._query_n, self._query_position,
                                           self._rigid._pos, self._rigid._vel, dt)
        
    @ti.func
    def _calculate_penetration_func(self, query_answer, query_n, tolerance_sdf: float):
        self._penetration.fill(0)
        query_n_max = maths.vec_max_func(query_n, self._batch_size)
        for batch_idx, sample_idx in ti.ndrange(self._batch_size, query_n_max):
            if sample_idx < query_n[batch_idx]:
                sdf_and_grad = query_answer[batch_idx, sample_idx]
                sdf = sdf_and_grad[0]
                if sdf < tolerance_sdf:
                    self._penetration[batch_idx] = 1
        
    @ti.kernel
    def _calculate_penetration_kernel(self, tolerance_sdf: float):
        self._generate_sample_on_vert_func(self._query_position, self._query_n)
        self._rigid._query_sdf_func(self._query_position, self._query_answer, self._query_n)
        self._generate_sample_on_face_func(self._query_position, self._query_n, self._query_answer, 
                                           self._face_sample_info, self._face_sample_cnt, self._sample_dx, tolerance_sdf)
        self._rigid._query_sdf_func(self._query_position, self._query_answer, self._query_n)
        self._calculate_penetration_func(self._query_answer, self._query_n, tolerance_sdf)

    def calculate_penetration(self, tolerance_sdf: float) -> torch.Tensor:
        """int, [B]"""
        self._calculate_penetration_kernel(tolerance_sdf)
        return self._penetration.to_torch(device=self._device)

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_penalty_force(self, dt: float, global_fatal_flag: torch.Tensor, **kwargs):
        self._calculate_penalty_force_kernel(dt)
        global_fatal_flag |= self._fatal_flag.to_torch(device=self._device)
        if self._log_verbose >= 1:
            logger.debug(f"{self._name} {self._query_n}")
        

@ti.data_oriented
class ClothArticulateForceCollision(ClothForceCollision):
    def __init__(self, cloth: Cloth, articulate: Articulate, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._articulate = articulate

        self._collisions: List[ClothRigidForceCollision] = []
        self._collision_map: Dict[str, ClothRigidForceCollision] = {}

        for rigid in self._articulate._collision_map.values():
            if rigid._sdf is not None:
                collision_cfg_modify = omegaconf.DictConfig(collision_cfg)
                collision_cfg_modify.name = collision_cfg.name + f"_{rigid._name}"
                collision_cfg_modify.store_sparse_matrix = False
                collision = ClothRigidForceCollision(cloth, rigid, collision_cfg_modify, global_cfg)
                self._collisions.append(collision)
                self._collision_map[rigid._name] = collision

        self._cloth_penalty_df_dt: ti.MatrixField = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._cloth._nv))
        """float, [B, V][3]"""

        self._cloth_hessian_matrix: ti.MatrixField = sim_utils.GLOBAL_CREATER.MatrixField(3, 3, float, shape=(self._batch_size, self._cloth._nf, 3, 3))
        """float, [B, F, 3, 3][3, 3]"""

        self._cloth_hessian_sparse = SparseMatrix(
            self._batch_size,
            self._cloth._nv * 3,
            self._cloth._nv * 3,
            self._cloth._nf * (9 ** 2),
            False,
        )

    @ti.kernel
    def _reset_collision_kernel(self):
        self._cloth_penalty_force.fill(0.)
        self._cloth_penalty_df_dt.fill(0.)
        self._cloth_hessian_matrix.fill(0.)

    @ti.kernel
    def _update_collision_kernel(self, collision: ti.template()):
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            self._cloth_penalty_force[batch_idx, vid] += collision._cloth_penalty_force[batch_idx, vid]
            self._cloth_penalty_df_dt[batch_idx, vid] += collision._cloth_penalty_df_dt[batch_idx, vid]
        for batch_idx, fid, i, j in ti.ndrange(self._batch_size, self._cloth._nf, 3, 3):
            self._cloth_hessian_matrix[batch_idx, fid, i, j] += collision._cloth_hessian_matrix[batch_idx, fid, i, j]
    
    @ti.kernel
    def _assemble_hessian_sparse_kernel(self):
        self._cloth_hessian_sparse.set_zero_func()
        for batch_idx, fid, i, j, m, n in ti.ndrange(self._batch_size, self._cloth._nf, 3, 3, 3, 3):
            vids = self._cloth._f2v[fid]
            self._cloth_hessian_sparse.add_value_func(
                batch_idx,
                vids[i] * 3 + m,
                vids[j] * 3 + n,
                self._cloth_hessian_matrix[batch_idx, fid, i, j][m, n]
            )

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_penalty_force(self, dt: float, global_fatal_flag: torch.Tensor, **kwargs):
        self._reset_collision_kernel()

        for collision in self._collisions:
            collision._calculate_penalty_force(dt=dt, global_fatal_flag=global_fatal_flag, **kwargs)
            self._update_collision_kernel(collision)

        self._assemble_hessian_sparse_kernel()

    @property
    def collision_map(self):
        return self._collision_map
