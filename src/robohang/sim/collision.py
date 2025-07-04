import taichi as ti

import torch

import omegaconf

from .sim_utils import BaseClass
from .cloth import Cloth
from .rigid import Rigid
from . import sim_utils


@ti.data_oriented
class ClothCollision(BaseClass):
    def __init__(self, cloth: Cloth, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(global_cfg)
        self._name: str = collision_cfg.name

        self._fatal_flag: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=bool, shape=(self._batch_size, ))
        """bool, [B, ]"""

        self._cloth = cloth
        self._dx_eps = self._cloth._dx_eps

        self._balance_distance: float = float(collision_cfg.balance_distance)

    @property
    def name(self):
        return self._name
    
    def get_fatal_flag(self) -> torch.Tensor:
        """int, [B, ]"""
        return self._fatal_flag.to_torch(self._device)


@ti.dataclass
class FaceSampleInfo:
    face_id: int
    bary_coor: ti.math.vec2

    def get_shape(self):
        return (3, )


@ti.data_oriented
class ClothRigidCollision(ClothCollision):
    def __init__(self, cloth: Cloth, rigid: Rigid, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> None:
        super().__init__(cloth=cloth, rigid=rigid, collision_cfg=collision_cfg, global_cfg=global_cfg, **kwargs)

        self._rigid = rigid

        self._max_collision: int = int(collision_cfg.max_collision)
        self._dv_eps: float = float(collision_cfg.dv_eps)
        assert self._max_collision >= self._cloth._nv
        assert self._dv_eps > 0.

        self._mu: ti.ScalarField = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """float, [B, ]"""
        self._mu.fill(collision_cfg.mu)

        self._query_position = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, self._max_collision))
        """float, [B, Q][3]"""
        self._query_answer = sim_utils.GLOBAL_CREATER.VectorField(n=4, dtype=float, shape=(self._batch_size, self._max_collision)) 
        """float, [B, Q][4]"""
        self._query_n = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))
        """int, [B, ]"""

    def set_mu(self, mu: torch.Tensor):
        self._mu.from_torch(mu)

    @ti.func
    def _generate_sample_on_vert_func(self, query_position, query_n):
        for batch_idx in range(self._batch_size):
            query_n[batch_idx] = self._cloth._nv
        for batch_idx, vid in ti.ndrange(self._batch_size, self._cloth._nv):
            query_position[batch_idx, vid] = self._cloth._pos[batch_idx, vid]
        
    @ti.kernel
    def _generate_sample_on_vert_kernel(self,
                                        query_position: ti.template(),
                                        query_n: ti.template(),):
        self._generate_sample_on_vert_func(query_position, query_n)

    @ti.func
    def _generate_sample_on_face_func(self, query_position, query_n, vert_sdf, face_sample_info, face_sample_cnt, sample_dx: float, balance_distance: float):
        self._fatal_flag.fill(False)
        face_sample_cnt.fill(0)

        for batch_idx in range(self._batch_size):
            query_n[batch_idx] = 0

        for batch_idx, fid in ti.ndrange(self._batch_size, self._cloth._nf):
            max_edge_length = self._cloth._get_face_max_length_func(batch_idx, fid)
            sample_res = ti.ceil(max_edge_length / sample_dx, dtype=int)

            vids = self._cloth._f2v[fid]
            pos0 = self._cloth._pos[batch_idx, vids[0]]
            pos1 = self._cloth._pos[batch_idx, vids[1]]
            pos2 = self._cloth._pos[batch_idx, vids[2]]

            maximum_sdf = max_edge_length + balance_distance

            if (vert_sdf[batch_idx, vids[0]][0] < maximum_sdf or
                vert_sdf[batch_idx, vids[1]][0] < maximum_sdf or
                vert_sdf[batch_idx, vids[2]][0] < maximum_sdf):

                for a in range(sample_res):
                    for b in range(sample_res - a):
                        c = sample_res - 1 - a - b
                        u, v, w = (a + 1 / 3) / sample_res, \
                            (b + 1 / 3) / sample_res, (c + 1 / 3) / sample_res

                        sample_idx = ti.atomic_add(query_n[batch_idx], +1)
                        if sample_idx < self._max_collision:
                            bc = ti.Vector([u, v], dt=float)
                            query_position[batch_idx, sample_idx] = pos0 * u + pos1 * v + pos2 * w
                            face_sample_info[batch_idx, sample_idx] = FaceSampleInfo(fid, bc)
                            face_sample_cnt[batch_idx, fid] += 1
                        else:
                            ti.atomic_add(query_n[batch_idx], -1)
                            old_fatal_flag = ti.atomic_or(self._fatal_flag[batch_idx], True)
                            if not old_fatal_flag:
                                print(f"[ERROR] batch_idx {batch_idx} rigid collision query number reaches maximum {self._max_collision}.")
        
    @ti.kernel
    def _generate_sample_on_face_kernel(self,
                                        query_position: ti.template(),
                                        query_n: ti.template(),
                                        vert_sdf: ti.template(),
                                        face_sample_info: ti.template(),
                                        face_sample_cnt: ti.template(),
                                        sample_dx: float,
                                        balance_distance: float):
        self._generate_sample_on_face_func(query_position, query_n, vert_sdf, face_sample_info, face_sample_cnt, sample_dx, balance_distance)