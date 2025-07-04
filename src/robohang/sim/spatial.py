import taichi as ti

import numpy as np
import torch

import omegaconf

from .sim_utils import BaseClass
from . import sim_utils
from .cloth import Cloth


@ti.dataclass
class BoundingBox:
    bounds: ti.types.matrix(2, 3, float)


@ti.func
def point_bounding_box_func(a: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = a
    bounds[1, :] = a
    return BoundingBox(bounds=bounds)


@ti.func
def line_bounding_box_func(a: ti.math.vec3, b: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(a, b)
    bounds[1, :] = ti.max(a, b)
    return BoundingBox(bounds=bounds)


@ti.func
def triangle_bounding_box_func(a: ti.math.vec3, b: ti.math.vec3, c: ti.math.vec3) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(a, b, c)
    bounds[1, :] = ti.max(a, b, c)
    return BoundingBox(bounds=bounds)


@ti.func
def merge_bounding_box_func(bb1: BoundingBox, bb2: BoundingBox) -> BoundingBox:
    bounds = ti.Matrix.zero(float, 2, 3)
    bounds[0, :] = ti.min(bb1.bounds[0, :], bb2.bounds[0, :])
    bounds[1, :] = ti.max(bb1.bounds[1, :], bb2.bounds[1, :])
    return BoundingBox(bounds=bounds)


@ti.data_oriented
class SpatialPartition(BaseClass):
    def __init__(self, spatial_partition_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)

        self._bounds = sim_utils.GLOBAL_CREATER.MatrixField(n=2, m=3, dtype=float, shape=())
        """float, [None][2, 3]"""
        self._bounds.from_torch(torch.tensor(spatial_partition_cfg.bounds, dtype=self._dtype, device=self._device))

        self._xyz_size = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=int, shape=())
        """int, [None][3, ]"""
        self._xyz_size.from_torch(torch.tensor(spatial_partition_cfg.xyz_size, dtype=self._dtype_int, device=self._device))
        self._max_spatial_cell_size = int(spatial_partition_cfg.max_spatial_cell_size)

        self._xyz_block_size = int(spatial_partition_cfg.xyz_block_size)
        self._spatial_cell_chunk_size = int(spatial_partition_cfg.spatial_cell_chunk_size)

        self._spatial_cell = ti.field(int)
        """int, [B, SX, SY, SZ, IL]"""
        self._spatial_pointer = ti.root.pointer(ti.ijkl, (
            self._batch_size,
            (spatial_partition_cfg.xyz_size[0] + self._xyz_block_size - 1) // self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[1] + self._xyz_block_size - 1) // self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[2] + self._xyz_block_size - 1) // self._xyz_block_size,
        ))
        self._spatial_pixel = self._spatial_pointer.dense(
            ti.ijkl,
            (1, self._xyz_block_size, self._xyz_block_size, self._xyz_block_size)
        )
        self._spatial_dynamic = self._spatial_pixel.dynamic(ti.axes(4), self._max_spatial_cell_size, chunk_size=self._spatial_cell_chunk_size)
        self._spatial_dynamic.place(self._spatial_cell)

        sim_utils.GLOBAL_CREATER.LogSparseField(shape=(
            self._batch_size,
            (spatial_partition_cfg.xyz_size[0] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[1] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            (spatial_partition_cfg.xyz_size[2] + self._xyz_block_size - 1) // self._xyz_block_size * self._xyz_block_size,
            self._max_spatial_cell_size,
        ))

        # for safety, prevent large serialized loop
        self._max_bb_occupy_num = int(spatial_partition_cfg.max_bb_occupy_num)
        self._fatal_flag: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))
        """bool, [B, ]"""
    
    def deactivate_all_kernel(self) -> torch.Tensor:
        """Clear fatal_flag and Return old fatal_flag"""
        self._spatial_pointer.deactivate_all()
        old_fatal_flag = self._fatal_flag.to_torch()
        self._fatal_flag.fill(0)
        return old_fatal_flag
    
    @ti.func
    def xyz2ijk_func(self, xyz: ti.math.vec3) -> ti.math.ivec3:
        bounds = self._bounds[None]
        xyz_size = self._xyz_size[None]

        return ti.math.clamp(
            ti.cast((xyz - bounds[0, :]) / 
                    (bounds[1, :] - bounds[0, :]) *
                    xyz_size, int),
            0, xyz_size - 1
        )

    @ti.func
    def add_bounding_box_func(self, batch_idx: int, bb: BoundingBox, value: int):
        lower_xyz = ti.min(bb.bounds[0, :], bb.bounds[1, :])
        upper_xyz = ti.max(bb.bounds[0, :], bb.bounds[1, :])
        lower_ijk = self.xyz2ijk_func(lower_xyz)
        upper_ijk = self.xyz2ijk_func(upper_xyz)
        cnt = calculate_loop_size_func(lower_ijk, upper_ijk)
        if cnt <= self._max_bb_occupy_num:
            for i, j, k in ti.ndrange(
                (lower_ijk[0], upper_ijk[0] + 1),
                (lower_ijk[1], upper_ijk[1] + 1),
                (lower_ijk[2], upper_ijk[2] + 1),
            ):
                self._spatial_cell[batch_idx, i, j, k].append(value)
        else:
            old_fatal_flag = ti.atomic_or(self._fatal_flag[batch_idx], True)
            if not old_fatal_flag:
                print(f"[ERROR] batch_idx {batch_idx} spatial loop size: {cnt} is too large")

    @ti.func
    def get_cell_length_func(self, batch_idx: int, i: int, j: int, k: int) -> int:
        return self._spatial_cell[batch_idx, i, j, k].length()
    
    @ti.func
    def get_cell_item_func(self, batch_idx: int, i: int, j: int, k: int, l: int) -> int:
        return self._spatial_cell[batch_idx, i, j, k, l]

    def get_fatal_flag(self) -> torch.Tensor:
        return self._fatal_flag.to_torch(self._device)


@ti.func
def calculate_loop_size_func(ijk_lower: ti.math.ivec3, ijk_upper: ti.math.ivec3) -> int:
    return (ijk_upper[0] - ijk_lower[0] + 1) * (ijk_upper[1] - ijk_lower[1] + 1) * (ijk_upper[2] - ijk_lower[2] + 1)