import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import Tuple

import torch

from .sparse import SparseMatrix
from . import maths
from . import sim_utils

@ti.data_oriented
class Modified_PCG_sparse_solver:
    def __init__(self, 
                 batch_size: int, 
                 nmax: int, 
                 nmax_triplet: int, 
                 max_iter: int, 
                 ) -> None:
        """Modified_PCG solver. solve Ax=b.
        A: n x n semi-positive definite matrix
        b: n
        """
        self.batch_size = batch_size
        
        assert nmax >= 1
        self.nmax = nmax
        self.n_field = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self.batch_size, ))
        self.n_field.fill(0)
        self.n_max_field = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=())
        self.n_max_field.fill(0)

        self.A = SparseMatrix(self.batch_size, nmax, nmax, nmax_triplet, True)
        self.A.set_zero_kernel()
        self.b = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.b.fill(0.0)
        self.eps_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.eps_f.fill(0.0)
        self.con_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.con_f.fill(0.0)

        self.block_dim_jacobi = 3
        self.num_block_jacobi = (nmax + 2) // 3
        self.p_block_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(
            self.batch_size, self.num_block_jacobi, 3, 3))
        """[B, BLOCK_NUM, 3, 3]"""
        self.p_block_f.fill(0.0)

        self.x_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.x_best_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.r_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.c_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.q_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.s_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.s_old_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.tmpvec1_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.tmpvec2_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, nmax))
        self.tmpsca1_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.tmpsca2_f = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))

        self.pos_ones_batch = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.pos_ones_batch.fill(+1.0)
        self.neg_ones_batch = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.neg_ones_batch.fill(-1.0)
        self.zeros_batch = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.zeros_batch.fill(0.0)

        self.delta_0 = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.delta_new = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.delta_old = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        self.delta_history = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, max_iter))
        """[B, ITER]"""

        self.delta_history_cnt = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=())
        self.delta_min = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))

        self.update_flag = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        """[B, ] float, 1.0 for update, 0.0 for stop"""
        self.update_flag.fill(1.0)

        self.restart_batch_mask = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self.batch_size, ))
        """[B, ] float, 1.0 for restart, 0.0 for stop"""

        self.iter_cnt = 0

    @ti.kernel
    def _calculate_block_jacobi_inverse_kernel(self, n_field: ti.template(), eps: float):
        block_dim = ti.static(self.block_dim_jacobi)
        max_n = maths.vec_max_func(n_field, self.batch_size)
        for b, i in ti.ndrange(self.batch_size, (max_n + block_dim - 1) // block_dim):
            if i < (n_field[b] + block_dim - 1) // block_dim:
                mat = ti.Matrix.zero(
                    float, block_dim, block_dim)
                for j, k in ti.static(ti.ndrange(block_dim, block_dim)):
                    jj = i * block_dim + j
                    kk = i * block_dim + k
                    if jj < n_field[b] and kk < n_field[b]:
                        mat[j, k] = self.p_block_f[b, i, j, k]

                u_mat, s_mat, v_mat = ti.svd(mat)
                s_inv_mat = ti.Matrix.zero(
                    float, block_dim, block_dim)
                is_singular = False
                for j in ti.static(range(block_dim)):
                    if s_mat[j, j] < eps:
                        s_inv_mat[j, j] = 0.0
                        is_singular = True
                    else:
                        s_inv_mat[j, j] = 1.0 / s_mat[j, j]

                mat_inv = ti.Matrix.zero(
                    float, block_dim, block_dim)
                if not is_singular:
                    mat_inv = v_mat @ s_inv_mat @ u_mat.transpose()
                else:
                    for j in ti.static(range(block_dim)):
                        if ti.abs(mat[j, j]) > eps:
                            mat_inv[j, j] = 1.0 / mat[j, j]
                        else:
                            mat_inv[j, j] = ti.math.sign(mat[j, j]) / eps

                for j, k in ti.static(ti.ndrange(block_dim, block_dim)):
                    self.p_block_f[b, i, j, k] = mat_inv[j, k]

    def init(self, n: int, A_eps: float):
        """init n, pre-cond and solver's variable."""
        self.n_field.fill(n)
        self.n_max_field.fill(n)
        self.A.compress(self.n_field, self.n_field, True)

        self.A.get_block_diag_kernel(self.p_block_f, self.block_dim_jacobi)
        self._calculate_block_jacobi_inverse_kernel(self.n_field, A_eps)

        self.iter_cnt = 0

    def _precond_mul_vec(self, batch_mask: ti.Field, ans: ti.Field, vec: ti.Field):
        """
        precond @ vec = ans
        """
        maths.block_mul_vec_batch_kernel(self.batch_size, batch_mask, ans, self.p_block_f, vec, self.n_field, self.block_dim_jacobi)

    @ti.kernel
    def _restart_stage_1_kernel(self, batch_mask: ti.template()):
        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
                self.x_f[batch_idx, i] = self.x_best_f[batch_idx, i]
        self.A.mul_vec_func(batch_mask, self.r_f, self.x_f, self.n_field)

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
                self.r_f[batch_idx, i] = self.con_f[batch_idx, i] * \
                    (self.b[batch_idx, i] - self.r_f[batch_idx, i])
                self.s_f[batch_idx, i] = 0.0
    
    @ti.kernel
    def _restart_stage_2_kernel(self, batch_mask: ti.template(), CG_relative_tol:float) -> bool:
        for batch_idx in range(self.batch_size):
            if batch_mask[batch_idx] != 0.0:
                self.delta_new[batch_idx] = 0.0

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and batch_mask[batch_idx] != 0.0:
                self.c_f[batch_idx, i] *= self.con_f[batch_idx, i]
                self.delta_new[batch_idx] += self.c_f[batch_idx, i] * self.r_f[batch_idx, i]

        all_is_converge = True
        for batch_idx in range(self.batch_size):
            if self.delta_new[batch_idx] < self.eps_f[batch_idx] or \
                self.delta_new[batch_idx] < self.delta_0[batch_idx] * CG_relative_tol ** 2:
                self.update_flag[batch_idx] = 0.0

            if self.update_flag[batch_idx] != 0.0:
                all_is_converge = False

        return all_is_converge
        
    def _restart(self, batch_mask: ti.Field, CG_relative_tol: float) -> bool:
        self._restart_stage_1_kernel(batch_mask)
        self._precond_mul_vec(batch_mask, self.c_f, self.r_f)
        return self._restart_stage_2_kernel(batch_mask, CG_relative_tol)

    def _iter_init(self, CG_relative_tol: float) -> bool:
        # calculate delta_0
        maths.vec_mul_vec_batch_kernel(self.batch_size, self.update_flag, self.tmpvec1_f, self.b, self.con_f, self.n_field)
        self._precond_mul_vec(self.update_flag, self.tmpvec2_f, self.tmpvec1_f)
        maths.vec_dot_vec_batch_kernel(self.batch_size, self.update_flag, self.tmpvec2_f, self.tmpvec1_f, self.delta_0, self.n_field)
        self.delta_min.copy_from(self.delta_0)
        return self._restart(self.update_flag, CG_relative_tol)

    @ti.kernel
    def _iter_step_stage_1_kernel(self) -> ti.types.vector(2, bool):
        """
        ```
        [PREV] q := A @ c (include)
        q := con * q
        c_dot_q := c^T @ q
        if c_dot_q > eps_f:
            alpha := delta_new / c_dot_q
            is_converge := False
        else:
            alpha := 0.0
            is_converge := False
        x := x + alpha * c
        r := r - alpha * q
        s_old := s
        return is_converge
        [NEXT] s := M^-1 @ r (M^-1 @ A ~ I)
        ```
        """
        self.A.mul_vec_func(self.pos_ones_batch, self.q_f, self.c_f, self.n_field)

        # tmpsca1_f = c_dot_q, tmpsca2_f = res
        self.tmpsca1_f.fill(0.0)
        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx]:
                self.q_f[batch_idx, i] *= self.con_f[batch_idx, i]
                self.tmpsca1_f[batch_idx] += self.c_f[batch_idx, i] * self.q_f[batch_idx, i]

        all_is_converge = True
        need_restart = False
        for batch_idx in range(self.batch_size):
            if self.tmpsca1_f[batch_idx] < self.eps_f[batch_idx]:
                self.update_flag[batch_idx] = 0.0
            # tmpsca2_f = alpha
            else:
                self.tmpsca2_f[batch_idx] = self.delta_new[batch_idx] / self.tmpsca1_f[batch_idx]
            
            if self.update_flag[batch_idx] != 0.0:
                all_is_converge = False
            if self.restart_batch_mask[batch_idx] != 0.0:
                need_restart = True

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and self.update_flag[batch_idx] != 0.0:
                self.x_f[batch_idx, i] += self.tmpsca2_f[batch_idx] * self.c_f[batch_idx, i]
                self.r_f[batch_idx, i] -= self.tmpsca2_f[batch_idx] * self.q_f[batch_idx, i]
                self.s_old_f[batch_idx, i] = self.s_f[batch_idx, i]
            
        return all_is_converge, need_restart

    @ti.kernel
    def _iter_step_stage_2_kernel(self, CG_relative_tol: float, restart_threshold: float) -> ti.types.vector(2, bool):
        """
        ```
        [PREV] s := M^-1 @ r (M^-1 @ A ~ I)
        delta_old := delta_new
        tmp1 := s - s_old
        delta_new := tmp1^T @ r
        beta := delta_new / max(delta_old, eps)
        if delta_new < delta_minimum:
            x_best := x
        c := s + beta * c
        c := con * c
        return delta_new < max(eps_f, delta_0 * rel_eps ^ 2) or plateau
        [NEXT] q := A @ c
        ```
        """
        old_cnt = ti.atomic_add(self.delta_history_cnt[None], 1)

        for batch_idx in range(self.batch_size):
            self.delta_old[batch_idx] = self.delta_new[batch_idx]
            self.delta_new[batch_idx] = 0.0

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx] and self.update_flag[batch_idx] != 0.0:
                self.delta_new[batch_idx] += (self.s_f[batch_idx, i] - self.s_old_f[batch_idx, i]) * self.r_f[batch_idx, i]

        for batch_idx in range(self.batch_size):
            if self.update_flag[batch_idx] != 0.0:
                self.delta_history[batch_idx, old_cnt] = self.delta_new[batch_idx]
            
            if self.delta_new[batch_idx] < self.delta_min[batch_idx]:
                self.delta_min[batch_idx] = self.delta_new[batch_idx]

            self.tmpsca1_f[batch_idx] = 0.0
            self.tmpsca2_f[batch_idx] = self.delta_new[batch_idx] / \
                ti.max(self.delta_old[batch_idx], self.eps_f[batch_idx]) # beta

        for batch_idx, i in ti.ndrange(self.batch_size, self.n_max_field[None]):
            if i < self.n_field[batch_idx]:
                if self.delta_min[batch_idx] == self.delta_new[batch_idx]:
                    self.x_best_f[batch_idx, i] = self.x_f[batch_idx, i]
                self.c_f[batch_idx, i] = (self.s_f[batch_idx, i] + self.tmpsca2_f[batch_idx] * self.c_f[batch_idx, i]) * self.con_f[batch_idx, i]
            
        all_is_converge = True
        need_restart = False
        for batch_idx in range(self.batch_size):
            if (self.delta_new[batch_idx] > self.delta_old[batch_idx] and
                self.delta_new[batch_idx] > restart_threshold * self.delta_min[batch_idx]):
                self.restart_batch_mask[batch_idx] = 1.0
                need_restart = True
            else:
                self.restart_batch_mask[batch_idx] = 0.0
            
            if (self.delta_new[batch_idx] < self.eps_f[batch_idx] or
                self.delta_new[batch_idx] < self.delta_0[batch_idx] * CG_relative_tol ** 2):
                self.update_flag[batch_idx] = 0.0

            if self.update_flag[batch_idx] != 0.0:
                all_is_converge = False
            
        return all_is_converge, need_restart

    @sim_utils.GLOBAL_TIMER.timer
    def _iter_step(self, CG_relative_tol: float, restart_threshold: float) -> Tuple[bool, bool]:
        all_is_converge, need_restart = self._iter_step_stage_1_kernel()
        if not all_is_converge:
            self._precond_mul_vec(self.pos_ones_batch, self.s_f, self.r_f)
            all_is_converge, need_restart = self._iter_step_stage_2_kernel(CG_relative_tol, restart_threshold)
            return all_is_converge, need_restart # converge = not update
        else:
            return True, False

    def _solve_python(self, torch_device, max_iter: int, CG_relative_tol: float, restart_threshold: float) -> torch.Tensor:
        assert max_iter >= 1

        self.update_flag.copy_from(self.pos_ones_batch)

        self.x_f.fill(0.0)
        self.x_best_f.fill(0.0)

        self.delta_history.fill(0.0)
        self.delta_history_cnt[None] = 0
        all_is_converge = self._iter_init(CG_relative_tol)

        _iter_cnt = 0

        if not all_is_converge:
            remain_iter = min(max_iter, int(self.n_max_field[None]))

            while remain_iter >= 1:
                all_is_converge, need_restart = self._iter_step(CG_relative_tol, restart_threshold)
                _iter_cnt += 1

                remain_iter -= 1
                self.iter_cnt += 1

                if (not all_is_converge) and need_restart:
                    all_is_converge = self._restart(self.restart_batch_mask, CG_relative_tol)

                if all_is_converge:
                    break
        
        logger.info(f"CG iter_cnt {_iter_cnt}")

        return self.x_best_f.to_torch(torch_device)

    @sim_utils.GLOBAL_TIMER.timer
    def solve(self, torch_device, max_iter: int, CG_relative_tol: float, restart_threshold: float = 1e1) -> torch.Tensor:
        """return the solution of Ax = b."""
        return self._solve_python(torch_device, max_iter, CG_relative_tol, restart_threshold)
