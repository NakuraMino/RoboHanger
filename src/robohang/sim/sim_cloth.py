import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import List

import torch

import omegaconf

from .sim_utils import BaseClass
from .cloth import Cloth
from .mpcg import Modified_PCG_sparse_solver
from .sparse import SparseMatrix
from .collision_force import ClothForceCollision
from .collision_position import ClothPositionCollision
from . import spatial
from . import ccd
from . import sim_utils


@ti.data_oriented
class ClothSim(BaseClass):
    def __init__(
        self, 
        cloth: Cloth,
        force_collisions: List[ClothForceCollision],
        position_collisions: List[ClothPositionCollision],
        cloth_sim_cfg: omegaconf.DictConfig,
        global_cfg: omegaconf.DictConfig
    ) -> None:
        super().__init__(global_cfg)

        self._cloth = cloth
        self._force_collisions = force_collisions
        self._position_collisions = position_collisions

        # total force
        self._total_force = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv * 3))
        """[B, V * 3]"""
        # total force's derivative
        self._total_df_dt = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv * 3))
        """[B, V * 3]"""
        self._self_intersection = sim_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))

        # total hessian
        nmax_triplet_sum = self._cloth._hessian_sparse.nmax_triplet
        for force_collision in self._force_collisions:
            nmax_triplet_sum += force_collision._cloth_hessian_sparse.nmax_triplet
        nmax_triplet_sum += self._cloth._nv * (3 ** 2) # external hessian

        self._total_hessian = SparseMatrix(
            batch_size=self._batch_size,
            nmax_row=self._cloth._nv * 3,
            nmax_column=self._cloth._nv * 3,
            nmax_triplet=nmax_triplet_sum, store_dense=False
        )
        
        # solver
        self._mass_vec: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv * 3))
        """float, [B, V * 3]"""
        self._vel_vec: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv * 3))
        """float, [B, V * 3]"""
        self._dot_vec: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv * 3))
        """float, [B, V * 3]"""
        self._tmpsca_1: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B, ]"""
        self._tmpsca_2: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B, ]"""
        self._nv_x_3: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))
        """int, [B, ]"""
        self._nv_x_3.fill(self._cloth._nv * 3)

        self._CG_max_iter: int = int(cloth_sim_cfg.CG.CG_max_iter)
        self._CG_relative_tol: float = float(cloth_sim_cfg.CG.CG_relative_tol)
        self._CG_dx_tol: float = float(cloth_sim_cfg.CG.CG_dx_tol)
        self._CG_restart_threshold: float = float(cloth_sim_cfg.CG.get("CG_restart_threshold", 1e1))
        self._solver = Modified_PCG_sparse_solver(
            batch_size=self._batch_size,
            nmax=self._cloth._nv * 3,
            nmax_triplet=self._cloth._nv * 3 + self._total_hessian.nmax_triplet,
            max_iter=self._CG_max_iter,
        )

        # ccd
        self._use_ccd: bool = bool(cloth_sim_cfg.use_ccd)
        if self._use_ccd:
            self._prev_max_step_length = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv))
            """float, [B, V], 0 ~ 1"""
            self._curr_max_step_length = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, self._cloth._nv))
            """float, [B, V], 0 ~ 1"""
            self._batch_step_length = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
            """float, [B, ]"""

            self._num_max_step_iterations: int = int(cloth_sim_cfg.ccd.num_max_step_iterations)
            self._num_cubic_solver_iterations: int = int(cloth_sim_cfg.ccd.num_cubic_solver_iterations)
            self._ccd_step_discount: float = float(cloth_sim_cfg.ccd.ccd_step_discount)
            self._min_step_length: float = float(cloth_sim_cfg.ccd.min_step_length)
            self._ccd_eps: float = self._cloth._dx_eps

        # collision_masks
        self._collision_mask_block_size: int = int(cloth_sim_cfg.collision_mask_block_size)

        self._vert_face_mask: ti.ScalarField
        """int, [B, V, F]"""
        self._vert_face_mask_pointer: ti.SNode

        self._edge_edge_mask: ti.ScalarField
        """int, [B, E, E]"""
        self._edge_edge_mask_pointer: ti.SNode
        
        self._build_collision_mask()

        # damping
        self._clamp_velocity_use = bool(cloth_sim_cfg.clamp_velocity.use)
        self._clamp_velocity_threshold = float(cloth_sim_cfg.clamp_velocity.threshold)
        self._clamp_velocity_strength = float(cloth_sim_cfg.clamp_velocity.strength)

        # misc
        self._skip = bool(cloth_sim_cfg.skip)
        if self._skip:
            print("[WARN] skip cloth simulation")

    def _build_collision_mask(self):
        bs = self._collision_mask_block_size
        self._vert_face_mask: ti.ScalarField = ti.field(int)
        self._vert_face_mask_pointer: ti.SNode = ti.root.pointer(
            ti.ijk,
            ((self._batch_size),
             (self._cloth._nv + bs - 1) // bs,
             (self._cloth._nf + bs - 1) // bs,)
            )
        self._vert_face_mask_pointer.bitmasked(ti.ijk, (1, bs, bs)).place(self._vert_face_mask)
        sim_utils.GLOBAL_CREATER.LogSparseField(shape=((self._batch_size),
                                                   (self._cloth._nv + bs - 1) // bs * bs,
                                                   (self._cloth._nf + bs - 1) // bs * bs))
        
        self._edge_edge_mask: ti.ScalarField = ti.field(int)
        self._edge_edge_mask_pointer: ti.SNode = ti.root.pointer(
            ti.ijk,
            ((self._batch_size),
             (self._cloth._ne + bs - 1) // bs,
             (self._cloth._ne + bs - 1) // bs,)
            )
        self._edge_edge_mask_pointer.bitmasked(ti.ijk, (1, bs, bs)).place(self._edge_edge_mask)
        sim_utils.GLOBAL_CREATER.LogSparseField(shape=(
            (self._batch_size),
            (self._cloth._ne + bs - 1) // bs * bs,
            (self._cloth._ne + bs - 1) // bs * bs
        ))

    @ti.kernel
    def _fill_total_force_df_dt(self, total_force: ti.template(), total_df_dt: ti.template(), gravity: ti.types.ndarray(dtype=ti.math.vec3)):
        total_df_dt.fill(0.0)
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            total_force[batch_idx, 3 * vid + j] = self._cloth._mass[batch_idx, vid] * gravity[batch_idx][j]

    @ti.kernel
    def _update_total_force_kernel(self, total_force: ti.template(), force: ti.template()):
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            total_force[batch_idx, 3 * vid + j] += force[batch_idx, vid][j]

    @ti.kernel
    def _update_total_force_df_dt_kernel(self, total_force: ti.template(), total_df_dt: ti.template(), force: ti.template(), df_dt: ti.template()):
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            total_force[batch_idx, 3 * vid + j] += force[batch_idx, vid][j]
            total_df_dt[batch_idx, 3 * vid + j] += df_dt[batch_idx, vid][j]
    
    @ti.kernel
    def _update_cloth_external_force_kernel(self):
        """add `self._cloth._external_force` to `self._total_force` and fill `self._cloth._external_force` with 0"""
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            self._total_force[batch_idx, vid * 3 + j] += self._cloth._external_force[batch_idx, vid][j]
        self._cloth._external_force.fill(0)
    
    @ti.kernel
    def _update_cloth_external_hessian_kernel(self):
        """add `self._cloth._external_hessian` to `self._total_hessian` and fill `self._cloth._external_hessian` with 0"""
        for batch_idx, vid, i, j in ti.ndrange(self._batch_size, self._cloth._nv, 3, 3):
            self._total_hessian.add_value_func(batch_idx, vid * 3 + i, vid * 3 + j, self._cloth._external_hessian[batch_idx, vid][i, j])
        self._cloth._external_hessian.fill(0)
    
    @ti.kernel
    def _init_CG_solver_A_b_cons_eps_kernel(self, dt: float, dx_tol: float) -> float:
        # (M + dt^2 * hess) @ dv = (f - hess @ v * dt) * dt + (df/dt) * dt^2

        # init A & eps
        avg_mass = 0.0
        self._solver.eps_f.fill(0.0)
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            self._mass_vec[batch_idx, 3 * vid + j] = self._cloth._mass[batch_idx, vid]
            self._solver.eps_f[batch_idx] += (dx_tol / dt) ** 2 * self._cloth._mass[batch_idx, vid]
            avg_mass += self._cloth._mass[batch_idx, vid]
        avg_mass /= (self._batch_size * self._cloth._nv * 3)

        for batch_idx in range(self._batch_size):
            self._tmpsca_1[batch_idx] = 1.0
            self._tmpsca_2[batch_idx] = dt ** 2

        self._solver.A.sparse_add_diag_func(
            self._mass_vec, 
            self._total_hessian, 
            self._tmpsca_1,
            self._tmpsca_2,
            self._nv_x_3,
        )

        # init constraint
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            if self._cloth._constraint[batch_idx, vid][j] != 0.0:
                self._solver.con_f[batch_idx, 3 * vid + j] = 1.0
            else:
                self._solver.con_f[batch_idx, 3 * vid + j] = 0.0

        # init b
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            self._vel_vec[batch_idx, 3 * vid + j] = self._cloth._vel[batch_idx, vid][j]

        self._total_hessian.mul_vec_func(
            self._solver.pos_ones_batch,
            self._dot_vec,
            self._vel_vec,
            self._nv_x_3,
        ) # hess @ v

        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            self._solver.b[batch_idx, 3 * vid + j] = (
                self._total_force[batch_idx, 3 * vid + j] * dt -
                self._dot_vec[batch_idx, 3 * vid + j] * dt ** 2 + 
                self._total_df_dt[batch_idx, 3 * vid + j] * dt ** 2
            )

        return avg_mass
    
    @ti.kernel
    def _update_cloth_velocity_kernel(self, dv: ti.types.ndarray()):
        for batch_idx, vid, j in ti.ndrange(self._batch_size, self._cloth._nv, 3):
            self._cloth._vel[batch_idx, vid][j] += dv[batch_idx, vid * 3 + j]

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_cloth_force(self):
        self._cloth._calculate_elastic_force()

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_collision_force(self, dt: float, spatial_partition: spatial.SpatialPartition, global_fatal_flag: torch.Tensor):
        for collision in self._force_collisions:
            collision._calculate_penalty_force(
                dt=dt, 
                spatial_partition=spatial_partition,
                global_fatal_flag=global_fatal_flag,
                vert_face_mask=self._vert_face_mask,
                vert_face_mask_pointer=self._vert_face_mask_pointer,
                edge_edge_mask=self._edge_edge_mask,
                edge_edge_mask_pointer=self._edge_edge_mask_pointer,
            )

    @sim_utils.GLOBAL_TIMER.timer
    def _assemble_total_force_df_dt(self, gravity: torch.Tensor):
        self._fill_total_force_df_dt(self._total_force, self._total_df_dt, gravity)
        self._update_cloth_external_force_kernel()
        self._update_total_force_kernel(self._total_force, self._cloth._elastic_force)
        for collision in self._force_collisions:
            if collision._cloth_penalty_df_dt is None:
                self._update_total_force_kernel(self._total_force, collision._cloth_penalty_force)
            else:
                self._update_total_force_df_dt_kernel(self._total_force, self._total_df_dt, collision._cloth_penalty_force, collision._cloth_penalty_df_dt)

    @sim_utils.GLOBAL_TIMER.timer
    def _assemble_total_hessian(self):
        self._total_hessian.set_zero_kernel()
        self._update_cloth_external_hessian_kernel()
        self._total_hessian.add_sparse_kernel(self._cloth._hessian_sparse)
        for collision in self._force_collisions:
            self._total_hessian.add_sparse_kernel(collision._cloth_hessian_sparse)

    @sim_utils.GLOBAL_TIMER.timer
    def _init_solver(self, dt: float):
        avg_mass = self._init_CG_solver_A_b_cons_eps_kernel(dt, self._CG_dx_tol)
        self._solver.init(self._cloth._nv * 3, avg_mass * sim_utils.get_eps(self._dtype))
    
    @sim_utils.GLOBAL_TIMER.timer
    def _call_solver(self) -> torch.Tensor:
        dv = self._solver.solve(self._device, self._CG_max_iter, self._CG_relative_tol, self._CG_restart_threshold)
        return dv

    @sim_utils.GLOBAL_TIMER.timer
    def _update_cloth_velocity(self, dv: torch.Tensor, dt: float):
        dv_list = [dv]
        for collision in self._position_collisions:
            dv_list.append(collision._calculate_collision_dv(dv, dt))
        self._update_cloth_velocity_kernel(sum(dv_list))

    @ti.kernel
    def _clamp_cloth_velocity_kernel(self, dt: float):
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            vel = self._cloth._vel[batch_idx, vid]
            vel_norm = ti.math.length(vel)
            if vel_norm > self._clamp_velocity_threshold:
                self._cloth._vel[batch_idx, vid] = (
                    self._clamp_velocity_threshold + 
                    ti.math.log(
                        (vel_norm - self._clamp_velocity_threshold) * 
                        self._clamp_velocity_strength + 1.
                    ) / self._clamp_velocity_strength
                ) * (vel / ti.max(vel_norm, self._cloth._dx_eps / dt))
    
    @sim_utils.GLOBAL_TIMER.timer
    def _clamp_cloth_velocity(self, dt: float):
        self._clamp_cloth_velocity_kernel(dt)

    @sim_utils.GLOBAL_TIMER.timer
    def _step_vel_cloth(self, dt: float, spatial_partition: spatial.SpatialPartition, global_fatal_flag: torch.Tensor, gravity: torch.Tensor):
        """
        Modify global_fatal_flag in place!
        """
        if self._skip:
            return
        
        assert gravity.shape == (self._batch_size, 3)

        self._calculate_cloth_force()
        self._calculate_collision_force(dt, spatial_partition, global_fatal_flag)

        self._assemble_total_force_df_dt(gravity)
        self._assemble_total_hessian()

        self._init_solver(dt)
        dv = self._call_solver()
        self._update_cloth_velocity(dv, dt)

        if self._clamp_velocity_use:
            self._clamp_cloth_velocity(dt)

    @ti.kernel
    def _step_pos_max_length_impl_kernel(self, dt: float):
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            self._cloth._pos[batch_idx, vid] += self._cloth._vel[batch_idx, vid] * dt

    @ti.kernel
    def _ccd_iter_init_kernel(self):
        self._curr_max_step_length.fill(1.)
        self._batch_step_length.fill(1.)
                
    @ti.func
    def _ccd_face_moving_bounding_box_func(self, batch_idx: int, dt: float, fid: int) -> spatial.BoundingBox:
        """
        `velocity = cloth._vel * self._prev_max_step_length`
        """
        vids = self._cloth._f2v[fid]
        x0 = self._cloth._pos[batch_idx, vids[0]]
        x1 = self._cloth._pos[batch_idx, vids[1]]
        x2 = self._cloth._pos[batch_idx, vids[2]]
        v0 = self._cloth._vel[batch_idx, vids[0]]
        v1 = self._cloth._vel[batch_idx, vids[1]]
        v2 = self._cloth._vel[batch_idx, vids[2]]
        b1 = spatial.triangle_bounding_box_func(x0, x1, x2)
        b2 = spatial.triangle_bounding_box_func(
            x0 + v0 * dt * self._prev_max_step_length[batch_idx, vids[0]],
            x1 + v1 * dt * self._prev_max_step_length[batch_idx, vids[1]],
            x2 + v2 * dt * self._prev_max_step_length[batch_idx, vids[2]],
        )
        return spatial.merge_bounding_box_func(b1, b2)
    
    @ti.func
    def _ccd_vert_moving_bounding_box_func(self, batch_idx: int, dt: float, vid: int) -> spatial.BoundingBox:
        """
        `velocity = cloth._vel * self._prev_max_step_length`
        """
        x = self._cloth._pos[batch_idx, vid]
        v = self._cloth._vel[batch_idx, vid]
        return spatial.line_bounding_box_func(
            x, 
            x + v * dt * self._prev_max_step_length[batch_idx, vid],
        )
    
    @ti.func
    def _ccd_edge_moving_bounding_box_func(self, batch_idx: int, dt: float, eid: int) -> spatial.BoundingBox:
        """
        `velocity = cloth._vel * self._prev_max_step_length`
        """
        vids = self._cloth._e2v[eid]
        x0 = self._cloth._pos[batch_idx, vids[0]]
        x1 = self._cloth._pos[batch_idx, vids[1]]
        v0 = self._cloth._vel[batch_idx, vids[0]]
        v1 = self._cloth._vel[batch_idx, vids[1]]
        b1 = spatial.line_bounding_box_func(x0, x1)
        b2 = spatial.line_bounding_box_func(
            x0 + v0 * dt * self._prev_max_step_length[batch_idx, vids[0]],
            x1 + v1 * dt * self._prev_max_step_length[batch_idx, vids[1]],
        )
        return spatial.merge_bounding_box_func(b1, b2)

    @ti.func
    def _ccd_vert_face_func(self, batch_idx: int, dt: float, vid: int, fid: int, vert_face_mask):
        if ((vid != self._cloth._f2v[fid]).all() and 
            (not ti.atomic_or(vert_face_mask[batch_idx, vid, fid], 1))):
            # vert_face_mask[batch_idx, vid, fid] = 1
            vids = ti.Vector([
                vid,
                self._cloth._f2v[fid][0],
                self._cloth._f2v[fid][1],
                self._cloth._f2v[fid][2],
            ], dt=int)

            t = ccd.vert_face_ccd_func(
                self._cloth._pos[batch_idx, vids[0]],
                self._cloth._pos[batch_idx, vids[1]],
                self._cloth._pos[batch_idx, vids[2]],
                self._cloth._pos[batch_idx, vids[3]],
                self._cloth._vel[batch_idx, vids[0]] * self._prev_max_step_length[batch_idx, vids[0]],
                self._cloth._vel[batch_idx, vids[1]] * self._prev_max_step_length[batch_idx, vids[1]],
                self._cloth._vel[batch_idx, vids[2]] * self._prev_max_step_length[batch_idx, vids[2]],
                self._cloth._vel[batch_idx, vids[3]] * self._prev_max_step_length[batch_idx, vids[3]],
                dt,
                self._ccd_eps,
                self._num_cubic_solver_iterations,
            )
            for i in ti.static(range(4)):
                if t < 1. and t >= 0.:
                    ti.atomic_min(self._curr_max_step_length[batch_idx, vids[i]], 
                                  t * self._ccd_step_discount * self._prev_max_step_length[batch_idx, vids[i]])

    @ti.func
    def _ccd_edge_edge_func(self, batch_idx: int, dt: float, e1id: int, e2id: int, edge_edge_mask):
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
         
            t = ccd.edge_edge_ccd_func(
                self._cloth._pos[batch_idx, vids[0]],
                self._cloth._pos[batch_idx, vids[1]],
                self._cloth._pos[batch_idx, vids[2]],
                self._cloth._pos[batch_idx, vids[3]],
                self._cloth._vel[batch_idx, vids[0]] * self._prev_max_step_length[batch_idx, vids[0]],
                self._cloth._vel[batch_idx, vids[1]] * self._prev_max_step_length[batch_idx, vids[1]],
                self._cloth._vel[batch_idx, vids[2]] * self._prev_max_step_length[batch_idx, vids[2]],
                self._cloth._vel[batch_idx, vids[3]] * self._prev_max_step_length[batch_idx, vids[3]],
                dt,
                self._ccd_eps,
                self._num_cubic_solver_iterations,
            )
            for i in ti.static(range(4)):
                if t < 1. and t >= 0.:
                    ti.atomic_min(self._curr_max_step_length[batch_idx, vids[i]], 
                                  t * self._ccd_step_discount * self._prev_max_step_length[batch_idx, vids[i]])

    @ti.kernel
    def _ccd_iter_step_vert_face_kernel(
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
            bb = self._ccd_face_moving_bounding_box_func(batch_idx, dt, fid)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_1[batch_idx], cnt)
            all_cnt_1[batch_idx] += cnt
            spatial_partition.add_bounding_box_func(batch_idx, bb, fid)

        # vert-face detection
        for batch_idx_, vid in ti.ndrange(good_batch_indices.shape[0], self._cloth._nv):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._ccd_vert_moving_bounding_box_func(batch_idx, dt, vid)
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
                        self._ccd_vert_face_func(batch_idx, dt, vid, fid, vert_face_mask)
            else:
                old_fatal_flag = ti.atomic_or(spatial_partition._fatal_flag[batch_idx], True)
                if not old_fatal_flag:
                    print(f"[ERROR] batch_idx {batch_idx} ccd vert-face loop size: {cnt} is too large")

    @ti.kernel
    def _ccd_iter_step_edge_edge_kernel(
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
            bb = self._ccd_edge_moving_bounding_box_func(batch_idx, dt, e1id)
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            cnt = spatial.calculate_loop_size_func(ijk_lower, ijk_upper)
            ti.atomic_max(max_cnt_1[batch_idx], cnt)
            all_cnt_1[batch_idx] += cnt
            spatial_partition.add_bounding_box_func(batch_idx, bb, e1id)

        # edge-edge detection
        for batch_idx_, e2id in ti.ndrange(good_batch_indices.shape[0], self._cloth._ne):
            batch_idx = good_batch_indices[batch_idx_]
            bb = self._ccd_edge_moving_bounding_box_func(batch_idx, dt, e2id)
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
                    (ijk_lower[2], ijk_upper[2] + 1),
                ):
                    ti.loop_config(serialize=True)
                    for l in range(spatial_partition.get_cell_length_func(batch_idx, i, j, k)):
                        e1id = spatial_partition.get_cell_item_func(batch_idx, i, j, k, l)
                        self._ccd_edge_edge_func(batch_idx, dt, e1id, e2id, edge_edge_mask)
            else:
                old_fatal_flag = ti.atomic_or(spatial_partition._fatal_flag[batch_idx], True)
                if not old_fatal_flag:
                    print(f"[ERROR] batch_idx {batch_idx} ccd edge-edge loop size: {cnt} is too large")

    @sim_utils.GLOBAL_TIMER.timer    
    def _ccd_iter_step(
        self, 
        dt: float, 
        spatial_partition: spatial.SpatialPartition, 
        global_fatal_flag: torch.Tensor,
        vert_face_mask: ti.ScalarField,
        vert_face_mask_pointer: ti.SNode,
        edge_edge_mask: ti.ScalarField,
        edge_edge_mask_pointer: ti.SNode,
    ):

        vf_max_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_all_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_max_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        vf_all_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_max_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_all_cnt_1 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_max_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        ee_all_cnt_2 = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        
        # store previous max step
        self._prev_max_step_length.copy_from(self._curr_max_step_length)

        vert_face_mask_pointer.deactivate_all()
        spatial_partition.deactivate_all_kernel()
        self._ccd_iter_step_vert_face_kernel(dt, spatial_partition, torch.where(global_fatal_flag == 0)[0].to(dtype=self._dtype_int), vert_face_mask, vf_max_cnt_1, vf_all_cnt_1, vf_max_cnt_2, vf_all_cnt_2)
        global_fatal_flag |= spatial_partition.get_fatal_flag()
        
        edge_edge_mask_pointer.deactivate_all()
        spatial_partition.deactivate_all_kernel()
        self._ccd_iter_step_edge_edge_kernel(dt, spatial_partition, torch.where(global_fatal_flag == 0)[0].to(dtype=self._dtype_int), edge_edge_mask, ee_max_cnt_1, ee_all_cnt_1, ee_max_cnt_2, ee_all_cnt_2)
        global_fatal_flag |= spatial_partition.get_fatal_flag()

        if self._log_verbose >= 2:
            logger.debug(f"ccd global_fatal_flag:{global_fatal_flag}")
            logger.debug(f"ccd vf_max_cnt_1:{vf_max_cnt_1}")
            logger.debug(f"ccd vf_all_cnt_1:{vf_all_cnt_1}")
            logger.debug(f"ccd vf_max_cnt_2:{vf_max_cnt_2}")
            logger.debug(f"ccd vf_all_cnt_2:{vf_all_cnt_2}")
            logger.debug(f"ccd ee_max_cnt_1:{ee_max_cnt_1}")
            logger.debug(f"ccd ee_all_cnt_1:{ee_all_cnt_1}")
            logger.debug(f"ccd ee_max_cnt_2:{ee_max_cnt_2}")
            logger.debug(f"ccd ee_all_cnt_2:{ee_all_cnt_2}")

    @ti.kernel
    def _ccd_iter_update_pos_kernel(
        self, 
        dt: float,
        global_fatal_flag: ti.types.ndarray(),
    ):  
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            if self._prev_max_step_length[batch_idx, vid] > 0.:
                ti.atomic_min(
                    self._batch_step_length[batch_idx],
                    self._curr_max_step_length[batch_idx, vid] / 
                    self._prev_max_step_length[batch_idx, vid]
                )
                
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            vert_step_length = (
                self._batch_step_length[batch_idx] * 
                self._prev_max_step_length[batch_idx, vid]
            )
            self._cloth._pos[batch_idx, vid] += self._cloth._vel[batch_idx, vid] * dt * vert_step_length

        for batch_idx in range(self._batch_size):
            batch_step_length = self._batch_step_length[batch_idx]
            if batch_step_length < self._min_step_length:
                old_fatal_flag = ti.atomic_or(global_fatal_flag[batch_idx], True)
                if not old_fatal_flag:
                    print(f"[ERROR] batch_idx {batch_idx} ccd batch_step_length {batch_step_length} is too small")

    def _step_pos_ccd(self, dt: float, spatial_partition: spatial.SpatialPartition, global_fatal_flag: torch.Tensor):
        self._ccd_iter_init_kernel()
        for _ in range(self._num_max_step_iterations):
            self._ccd_iter_step(
                dt, spatial_partition,
                global_fatal_flag,
                self._vert_face_mask,
                self._vert_face_mask_pointer,
                self._edge_edge_mask,
                self._edge_edge_mask_pointer,
            )
        self._ccd_iter_update_pos_kernel(dt, global_fatal_flag)
                    
    def _step_pos_cloth(self, dt: float, spatial_partition: spatial.SpatialPartition, global_fatal_flag: torch.Tensor):
        """
        Modify global_fatal_flag in place!
        """
        if self._skip:
            return
        
        if not self._use_ccd:
            self._step_pos_max_length_impl_kernel(dt)
        else:
            self._step_pos_ccd(dt, spatial_partition, global_fatal_flag)

    @ti.func
    def _detect_self_intersection_face_edge_func(self, batch_idx: int, fid: int, eid: int):
        v0id, v1id, v2id = self._cloth._f2v[fid]
        v3id, v4id = self._cloth._e2v[eid]
        edge_on_face = ((v3id == v0id or v3id == v1id or v3id == v2id) or 
                        (v4id == v0id or v4id == v1id or v4id == v2id))
        if not edge_on_face:
            mat = ti.Matrix.zero(ti.f64, 3, 3)
            mat[:, 0] = ti.cast(self._cloth._pos[batch_idx, v3id] - self._cloth._pos[batch_idx, v4id], ti.f64)
            mat[:, 1] = ti.cast(self._cloth._pos[batch_idx, v1id] - self._cloth._pos[batch_idx, v0id], ti.f64)
            mat[:, 2] = ti.cast(self._cloth._pos[batch_idx, v2id] - self._cloth._pos[batch_idx, v0id], ti.f64)

            xyz_scale = ti.abs(mat).sum() / 9
            mat_det = mat.determinant()
            if ti.abs(mat_det) > (xyz_scale ** 2) * self._cloth._dx_eps:
                
                right = self._cloth._pos[batch_idx, v3id] - self._cloth._pos[batch_idx, v0id]
                left = mat.inverse() @ ti.cast(right, ti.f64)

                a, b, c = 1. - left[1] - left[2], left[1], left[2]
                t = left[0]
                abct = ti.Vector([a, b, c, t], ti.f64)
                zero_f64 = ti.cast(0.0, ti.f64)
                one_f64 = ti.cast(1.0, ti.f64)
                if (zero_f64 < abct).all() and (abct < one_f64).all():
                    self._self_intersection[batch_idx] = True

    @ti.kernel
    def _detect_self_intersection_kernel(self, spatial_partition: ti.template()):
        # set False
        self._self_intersection.fill(False)

        # spatial data structure
        for batch_idx, fid in ti.ndrange(self._batch_size, self._cloth._nf):
            spatial_partition.add_bounding_box_func(
                batch_idx, 
                spatial.triangle_bounding_box_func(
                    self._cloth._pos[batch_idx, self._cloth._f2v[fid][0]],
                    self._cloth._pos[batch_idx, self._cloth._f2v[fid][1]],
                    self._cloth._pos[batch_idx, self._cloth._f2v[fid][2]],
                ),
                fid,
            )

        # face-edge detection
        for batch_idx, eid in ti.ndrange(self._batch_size, self._cloth._ne):
            bb = spatial.line_bounding_box_func(
                self._cloth._pos[batch_idx, self._cloth._e2v[eid][0]],
                self._cloth._pos[batch_idx, self._cloth._e2v[eid][1]],
            )
            ijk_lower = spatial_partition.xyz2ijk_func(bb.bounds[0, :])
            ijk_upper = spatial_partition.xyz2ijk_func(bb.bounds[1, :])
            ti.loop_config(serialize=True)
            for i, j, k in ti.ndrange(
                (ijk_lower[0], ijk_upper[0] + 1),
                (ijk_lower[1], ijk_upper[1] + 1),
                (ijk_lower[2], ijk_upper[2] + 1),
            ):
                ti.loop_config(serialize=True)
                for l in range(spatial_partition.get_cell_length_func(batch_idx, i, j, k)):
                    fid = spatial_partition.get_cell_item_func(batch_idx, i, j, k, l)
                    self._detect_self_intersection_face_edge_func(batch_idx, fid, eid)

    def _detect_self_intersection(self, spatial_partition: spatial.SpatialPartition) -> torch.Tensor:
        """return int, [B, ]"""
        spatial_partition.deactivate_all_kernel()
        self._detect_self_intersection_kernel(spatial_partition)
        return self._self_intersection.to_torch(self._device)
    