import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import List, Union

import torch
import numpy as np
import trimesh

import omegaconf

from .sim_utils import BaseClass
from . import maths
from . import sim_utils

@ti.func
def ijk_to_xyz_func(ijk, bounds, size):
    """
    Args
        - ijk: [3]
        - bounds: [2, 3]
        - size: [3]
    Return:
        - xyz: [3]
    """
    return (bounds[1, :] * ijk + bounds[0, :] * (size - 1 - ijk)) / (size - 1)


@ti.func
def xyz_to_ijk_func(xyz, bounds, size):
    """
    Args
        - xyz: [3]
        - bounds: [2, 3]
        - size: [3]

    Return:
        - ijk: [3]
    """
    return (xyz - bounds[0, :]) / (bounds[1, :] - bounds[0, :]) * (size - 1)


@ti.data_oriented
class Sdf(BaseClass):
    def __init__(self, mesh: trimesh.Trimesh, sdf_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, name: str) -> None:
        assert isinstance(mesh, trimesh.Trimesh)
        super().__init__(global_cfg)

        self._eps = sim_utils.get_eps(self._dtype)

        self._sdf_cfg = omegaconf.DictConfig(sdf_cfg)
        if not hasattr(self._sdf_cfg, "size"):
            self._sdf_cfg.size = np.ceil(
                (mesh.bounds[1, :] - mesh.bounds[0, :] + 
                 self._sdf_cfg.expand_distance * 2) / np.array(self._sdf_cfg.diff)
            ).astype(int).tolist()
            print(f"'size' attribute not found, use 'diff={self._sdf_cfg.diff}', 'expand={self._sdf_cfg.expand_distance}' to calculate sdf's size {self._sdf_cfg.size}")

        self._reference_points_num = self._sdf_cfg.get("reference_points_num", 16)
        self._max_recalculate = self._sdf_cfg.get("max_recalculate", 5)
        self._max_sdf = float(self._sdf_cfg.get("max_sdf", 1e6))

        self._size_torch = torch.zeros((3, ), dtype=self._dtype_int, device=self._device)
        self._size_torch[...] = torch.tensor(self._sdf_cfg.size)
        assert (self._size_torch >= 2).all()

        self._size = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=int, shape=())
        """int, [None][3, ]"""
        self._size.from_torch(self._size_torch)
        self._sdf = sim_utils.GLOBAL_CREATER.VectorField(n=4, dtype=float, shape=(self._size_torch[0], self._size_torch[1], self._size_torch[2]))
        """float, [SX, SY, SZ, 4]"""
        self._tmp_sdf = sim_utils.GLOBAL_CREATER.VectorField(n=4, dtype=float, shape=(self._size_torch[0], self._size_torch[1], self._size_torch[2]))
        """float, [SX, SY, SZ, 4]"""

        self._count_through_faces = torch.zeros([*self._size_torch] + [self._reference_points_num], dtype=self._dtype_int, device=self._device)
        """float, [SX, SY, SZ, R]"""

        self._vertices = torch.tensor(mesh.vertices, dtype=self._dtype, device=self._device)
        """float, [V, 3]"""

        self._faces = torch.tensor(mesh.faces, dtype=self._dtype_int, device=self._device)
        """int, [F, 3]"""

        self._is_watertight = bool(mesh.is_watertight)

        if not self._is_watertight:
            self._max_recalculate = 1
            s = f"[WARN] when calculating sdf {name}, the mesh is not waterwight, thus the answer is not reliable."
            print(s), logger.warn(s)

        self._bounds_torch = torch.tensor(
            np.array(self._expand_bounds(mesh.bounds, expand_distance=self._sdf_cfg.expand_distance)),
            dtype=self._dtype, device=self._device
        )

        self._bounds = sim_utils.GLOBAL_CREATER.MatrixField(n=2, m=3, dtype=float, shape=())
        """float, [None][2, 3]"""
        self._bounds.from_torch(self._bounds_torch)

        self._calculate_sdf()

    def _expand_bounds(self, bounds: np.ndarray, expand_distance=0.1):
        """bounds: [2, 3]"""
        expanded_bounds = bounds.copy()
        expanded_bounds[0, :] -= expand_distance
        expanded_bounds[1, :] += expand_distance
        return expanded_bounds
    
    @ti.kernel
    def _calculate_sdf_impl_kernel(
        self, 
        vertices: ti.types.ndarray(dtype=ti.math.vec3),
        faces: ti.types.ndarray(dtype=ti.math.ivec3),
        count_through_faces: ti.types.ndarray(dtype=int),
        reference_points: ti.types.ndarray(dtype=ti.math.vec3),
        on_surface_threshold: float,
    ) -> bool:
        """
        Args
            - vertices: [V][3]
            - faces: [F][3]
            - count_through_faces: [SX, SY, SZ, R]
            - reference_points: [R][3]
        """
        bounds_mat = self._bounds[None]
        size_vec = self._size[None]

        R = reference_points.shape[0]
        SX, SY, SZ = size_vec
        F = faces.shape[0]

        # calculate unsigned distance
        for i, j, k in ti.ndrange(SX, SY, SZ):
            self._tmp_sdf[i, j, k][0] = +ti.math.inf
        for i, j, k, f in ti.ndrange(SX, SY, SZ, F):
            query_point = ijk_to_xyz_func(ti.Vector([i, j, k]), bounds_mat, size_vec)
            a = vertices[faces[f][0]]
            b = vertices[faces[f][1]]
            c = vertices[faces[f][2]]
            l, u, v, w = maths.get_distance_to_triangle_func(query_point, a, b, c, self._eps)
            ti.atomic_min(self._tmp_sdf[i, j, k][0], l)
        for i, j, k, f in ti.ndrange(SX, SY, SZ, F):
            query_point = ijk_to_xyz_func(ti.Vector([i, j, k]), bounds_mat, size_vec)
            a = vertices[faces[f][0]]
            b = vertices[faces[f][1]]
            c = vertices[faces[f][2]]
            l, u, v, w = maths.get_distance_to_triangle_func(query_point, a, b, c, self._eps)
            if l == self._tmp_sdf[i, j, k][0]:
                self._tmp_sdf[i, j, k][1:] = maths.safe_normalized_func(query_point - (u * a + v * b + w * c), self._eps)

        # determine sign
        for i, j, k, r in ti.ndrange(SX, SY, SZ, R):
            count_through_faces[i, j, k, r] = 0
        for i, j, k, f in ti.ndrange(SX, SY, SZ, F):
            query_point = ijk_to_xyz_func(ti.Vector([i, j, k]), bounds_mat, size_vec)
            mat = ti.Matrix.zero(ti.float64, 3, 3)
            mat[:, 1] = ti.cast(vertices[faces[f][1]], ti.f64) - ti.cast(vertices[faces[f][0]], ti.f64)
            mat[:, 2] = ti.cast(vertices[faces[f][2]], ti.f64) - ti.cast(vertices[faces[f][0]], ti.f64)
            right = ti.cast(query_point, ti.f64) - ti.cast(vertices[faces[f][0]], ti.f64)
            for r in range(reference_points.shape[0]):
                reference_point = reference_points[r]
                mat[:, 0] = ti.cast(query_point, ti.f64) - ti.cast(reference_point, ti.f64)
                left = mat.inverse() @ ti.cast(right, ti.f64)
                a, b, c = 1. - left[1] - left[2], left[1], left[2]
                t = left[0]
                zero_f64 = ti.cast(0.0, ti.f64)
                one_f64 = ti.cast(1.0, ti.f64)
                if t > zero_f64 and zero_f64 < a and a < one_f64 \
                    and zero_f64 < b and b < one_f64 \
                    and zero_f64 < c and c < one_f64:
                    count_through_faces[i, j, k, r] += 1

        success = True
        # calculate final result
        for i, j, k in ti.ndrange(SX, SY, SZ):
            odd_cnt = 0
            even_cnt = 0
            for r in range(R):
                if count_through_faces[i, j, k, r] % 2 == 0:
                    even_cnt += 1
                else:
                    odd_cnt += 1
            if odd_cnt != 0 and even_cnt != 0:
                if ti.abs(self._tmp_sdf[i, j, k][0]) > on_surface_threshold:
                    success = False
            sign = 1.0
            if odd_cnt > even_cnt:
                sign = -1.0
            self._tmp_sdf[i, j, k] *= sign
        return success

    @sim_utils.GLOBAL_TIMER.timer
    def _calculate_sdf(self):
        assert self._size_torch[0] * self._size_torch[1] * self._size_torch[2] * self._reference_points_num < sim_utils.MAX_RANGE
        assert self._size_torch[0] * self._size_torch[1] * self._size_torch[2] * self._faces.shape[0] < sim_utils.MAX_RANGE
        for recalculate_times in range(self._max_recalculate):
            _reference_points_xyz = torch.rand((self._reference_points_num, 3), 
                                                dtype=self._dtype, device=self._device)
            success = self._calculate_sdf_impl_kernel(
                self._vertices, 
                self._faces,
                self._count_through_faces,
                self._bounds_torch[1, :] * _reference_points_xyz + 
                self._bounds_torch[0, :] * (1. - _reference_points_xyz),
                1e-4,
            )
            if success:
                break
            else:
                print(f"calculate sdf failed... recalculate {recalculate_times}")

            if recalculate_times == self._max_recalculate - 1:
                print(f"calculate sdf failed... reach max recalculate number")
        
        self._sdf.copy_from(self._tmp_sdf)

    def recalculate_sdf(self, vertices: torch.Tensor, faces: torch.Tensor):
        self._vertices[...] = vertices
        self._faces[...] = faces
        self._calculate_sdf()

    @ti.func
    def _query_single_sdf_func(self,
                               position: ti.math.vec3,
                               fwd_mat: ti.math.mat4,
                               inv_mat: ti.math.mat4,
                               scale: float,
                               ) -> ti.math.vec4:
        answer = ti.Vector([self._max_sdf, 0., 0., 0.])
        bounds_mat = self._bounds[None]
        size_vec = self._size[None]
        
        pos3D_local = (inv_mat[:3, :3] @ position + inv_mat[:3, 3]) / scale
        if ((bounds_mat[0, :] <= pos3D_local) * (pos3D_local <= bounds_mat[1, :])).all():
            ijk = xyz_to_ijk_func(pos3D_local, bounds_mat, size_vec)
            ijk_integer = ti.math.clamp(
                ti.floor(ijk, int),
                0, size_vec - 2,
            )
            ijk_decimal = ti.math.clamp(
                ijk - ti.cast(ijk_integer, float),
                0., 1.,
            )

            # extract sdf val
            sdf_val = ti.Vector.zero(dt=ti.f32, n=8, m=4)
            for dx, dy, dz in ti.static(ti.ndrange(2, 2, 2)):
                sdf_val[dx * 4 + dy * 2 + dz, :] = self._sdf[ijk_integer[0] + dx, ijk_integer[1] + dy, ijk_integer[2] + dz][:]
            # trilinear interpolation
            sdf4D_local = maths.trilinear_4D_func(ijk_decimal, sdf_val)

            # write answer
            answer[0] = sdf4D_local[0] * scale
            answer[1:] = maths.safe_normalized_func(fwd_mat[:3, :3] @ sdf4D_local[1:], self._eps)

        return answer
    
    @ti.func
    def _query_sdf_func(self, position, fwd_mat, inv_mat, scale, answer, query_n):
        max_n = maths.vec_max_func(query_n, self._batch_size)
        for batch_idx, q in ti.ndrange(self._batch_size, max_n):
            if q < query_n[batch_idx]:
                answer[batch_idx, q] = self._query_single_sdf_func(
                    position[batch_idx, q],
                    fwd_mat[batch_idx],
                    inv_mat[batch_idx],
                    scale[batch_idx],
                )

    @ti.kernel
    def _query_sdf_kernel(self,
                          position: ti.template(),
                          fwd_mat: ti.template(),
                          inv_mat: ti.template(),
                          scale: ti.template(),
                          answer: ti.template(),
                          query_n: ti.template(),
                          ):
        """
        Args:
            - position: [B, Q][3]
            - fwd_mat: [B][4, 4]
            - inv_mat: [B][4, 4]
            - scale: [B]
            - answer: [B, Q][4]
            - query_n: [B]
        """
        self._query_sdf_func(position, fwd_mat, inv_mat, scale, answer, query_n)
            
    def _query_sdf(self, 
                   position: ti.Field, 
                   fwd_mat: ti.Field, 
                   inv_mat: ti.Field, 
                   scale: ti.Field, 
                   answer: ti.Field, 
                   query_n: ti.Field):
        assert position.shape[0] * position.shape[1] < sim_utils.MAX_RANGE
        self._query_sdf_kernel(position, fwd_mat, inv_mat, scale, answer, query_n)

    def query_sdf(self, 
                  position: ti.Field, 
                  fwd_mat: ti.Field, 
                  inv_mat: ti.Field, 
                  scale: ti.Field, 
                  answer: ti.Field, 
                  query_n: ti.Field):
        """query sdf and save answers in `answer`"""
        B, N = position.shape
        B_, N_ = answer.shape
        B__, = query_n.shape
        assert B == B_ and B == B__, f"{position.shape} {fwd_mat.shape} {inv_mat.shape} {answer.shape} {query_n.shape}"

        B, = fwd_mat.shape
        B_, = inv_mat.shape
        assert B == B_ and B == B__, f"{position.shape} {fwd_mat.shape} {inv_mat.shape} {answer.shape} {query_n.shape}"

        assert scale.shape == (B, ), f"{scale.shape}"
        self._query_sdf(position, fwd_mat, inv_mat, scale, answer, query_n)