import taichi as ti

from typing import Any, List, Optional, Union, Literal
import torch
import numpy as np
import copy

import trimesh
import trimesh.transformations as tra

import omegaconf

from .sim_utils import BaseClass
from . import sim_utils
from . import so3
from . import maths
from . import sdf


@ti.data_oriented
class Rigid(BaseClass):
    """Base class for all rigid bodies."""

    _torch_state_str = ["inertia", "mass", "inertial_origin"]
    _field_state_str = ["pos", "vel", "scale"]
    def __init__(self, rigid_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(global_cfg)
        self._name: str = rigid_cfg.name

        self._pos = sim_utils.GLOBAL_CREATER.VectorField(n=7, dtype=float, shape=(self._batch_size))
        """[B, 7] = 3 translation + 4 quaternion [w, x, y, z]"""

        self._vel = sim_utils.GLOBAL_CREATER.VectorField(n=6, dtype=float, shape=(self._batch_size))
        """[B, 6] = 3 linear + 3 angular"""

        self._scale = sim_utils.GLOBAL_CREATER.ScalarField(dtype=float, shape=(self._batch_size, ))
        """[B, ]"""
        self._scale.fill(1.)

        self._sdf: Optional[sdf.Sdf] = None
        self._fwd = sim_utils.GLOBAL_CREATER.MatrixField(n=4, m=4, dtype=float, shape=(self._batch_size, ))
        """float, [B, ][4, 4]"""
        self._inv = sim_utils.GLOBAL_CREATER.MatrixField(n=4, m=4, dtype=float, shape=(self._batch_size, ))
        """float, [B, ][4, 4]"""

        self._mesh: trimesh.Trimesh = None
        self._point_on_surface_n: int = 0
        self._point_on_surface: Optional[ti.MatrixField] = None
        """float, [P][3]"""
        self._point_on_surface_query: Optional[ti.MatrixField] = None
        """float, [B, P][3]"""
        self._point_on_surface_answer: Optional[ti.MatrixField] = None
        """float, [B, P][4]"""
        self._point_on_surface_query_n: Optional[ti.MatrixField] = None
        """int, [B]"""

        self._inertia: torch.Tensor = torch.zeros((self._batch_size, 3, 3), dtype=self._dtype, device=self._device)
        """float, [B, 3, 3]"""
        self._default_inertia = copy.deepcopy(global_cfg.default_inertia)
        self._inertia[...] = torch.diag(torch.tensor([self._default_inertia] * 3))[None, ...]
        
        self._mass: torch.Tensor = torch.zeros((self._batch_size, ), dtype=self._dtype, device=self._device)
        """float, [B, ]"""
        self._default_mass = copy.deepcopy(global_cfg.default_mass)
        self._mass[...] = self._default_mass

        self._inertial_origin: torch.Tensor = torch.zeros((self._batch_size, 4, 4), dtype=self._dtype, device=self._device)
        """float, [B, 4, 4]"""
        self._inertial_origin[...] = torch.eye(4)[None, ...]

        self._wrench: torch.Tensor = torch.zeros((self._batch_size, 6), dtype=self._dtype, device=self._device)
        """float, [B, 6], force*3 + torque*3 with respect to mass center in world frame, does not include gravity."""
    
    def get_pos(self) -> torch.Tensor:
        """[B, 7]"""
        return self._pos.to_torch(self._device)
    
    def set_pos(self, pos: torch.Tensor):
        """[B, 7]"""
        self._pos.from_torch(pos)

    def get_vel(self) -> torch.Tensor:
        """[B, 6]"""
        return self._vel.to_torch(self._device)

    def set_vel(self, vel: torch.Tensor):
        """[B, 6]"""
        self._vel.from_torch(vel)
    
    def get_scale(self) -> torch.Tensor:
        """[B, ]"""
        return self._scale.to_torch(self._device)

    def set_scale(self, scale: torch.Tensor):
        """[B, ]"""
        self._scale.from_torch(scale)

    def get_inertia(self) -> torch.Tensor:
        """[B, 3, 3]"""
        return self._inertia.clone()
        
    def set_inertia(self, inertia: torch.Tensor):
        """[B, 3, 3]"""
        self._inertia[...] = inertia
        self._inertia[...] = (self._inertia[...] + self._inertia[...].transpose(1, 2)) / 2. # force symmetry

    def get_mass(self) -> Union[None, torch.Tensor]:
        """[B, ]"""
        return self._mass.clone()
        
    def set_mass(self, mass: torch.Tensor):
        """[B, ]"""
        self._mass[...] = mass

    def get_inertial_origin(self) -> Union[None, torch.Tensor]:
        """[B, 4, 4]"""
        return self._inertial_origin.clone()
        
    def set_inertial_origin(self, inertial_origin: torch.Tensor):
        """[B, 4, 4]"""
        self._inertial_origin[...] = inertial_origin

    def _center_mass_pos_impl(self) -> torch.Tensor:
        return so3.matrix_to_pos7d(so3.pos7d_to_matrix(self._pos.to_torch(self._device)) @ self._inertial_origin)

    def center_mass_pos(self) -> torch.Tensor:
        """[B, 7]"""
        return self._center_mass_pos_impl()
    
    def _center_mass_vel_impl(self) -> torch.Tensor:
        vel_cm = self._vel.to_torch(self._device)
        pos_rel = (so3.pos7d_to_matrix(self._pos.to_torch(self._device))[:, :3, :3] @ self._inertial_origin[:, :3, [3]]).squeeze(2)
        vel_cm[:, :3] += vel_cm[:, 3:].cross(pos_rel, dim=1)
        return vel_cm
    
    def center_mass_vel(self) -> torch.Tensor:
        """[B, 6]"""
        return self._center_mass_vel_impl()

    def _linear_momentum_impl(self) -> torch.Tensor:
        return self._center_mass_vel_impl()[:, :3] * self._mass[:, None]
    
    def linear_momentum(self) -> torch.Tensor:
        """[B, 3]"""
        return self._linear_momentum_impl()

    def _angular_momentum_impl(self) -> torch.Tensor:
        pos_cm = self._center_mass_pos_impl()
        mat_cm = so3.pos7d_to_matrix(pos_cm)
        inertia_world = mat_cm[:, :3, :3] @ self._inertia @ mat_cm[:, :3, :3].transpose(1, 2)
        angular_momentum = (inertia_world @ self._vel.to_torch(self._device)[:, 3:, None]).squeeze(2)
        return angular_momentum
    
    def augular_momentum(self) -> torch.Tensor:
        """[B, 3]"""
        return self._angular_momentum_impl()
    
    def _set_wrench(self, wrench: torch.Tensor):
        self._wrench[...] = wrench
    
    def _reinit_sdf(self, sdf_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        self._sdf = sdf.Sdf(self._mesh, sdf_cfg, global_cfg, self._name)

    def reinit_sdf(self, sdf_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        self._reinit_sdf(sdf_cfg, global_cfg)

    def recalculate_sdf(self, vert: torch.Tensor, face: torch.Tensor):
        self._sdf.recalculate_sdf(vert, face)

    @ti.func
    def _query_sdf_func(self, position, answer, query_n):
        for batch_idx in range(self._batch_size):
            self._fwd[batch_idx] = so3.pos7d_to_matrix_func(self._pos[batch_idx])
            self._inv[batch_idx] = so3.inverse_transform_func(self._fwd[batch_idx])
        self._sdf._query_sdf_func(position, self._fwd, self._inv, self._scale, answer, query_n)

    @ti.kernel
    def _query_sdf_kernel(self, position: ti.template(), answer: ti.template(), query_n: ti.template()):
        self._query_sdf_func(position, answer, query_n)
            
    def query_sdf(self, position: ti.Field, answer: ti.Field, query_n: ti.Field):
        """
        Args:
            - position: [B, Q][3]
            - answer: [B, Q][4]
            - query_n: [B]
        """
        self._query_sdf_kernel(position, answer, query_n)

    def _sample_on_surface(self, count: int, fps_multiplier=10):
        assert (self._point_on_surface is None and 
                self._point_on_surface_query is None and 
                self._point_on_surface_answer is None and
                self._point_on_surface_query_n is None)
        count = int(count)
        self._point_on_surface = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(count, ))
        self._point_on_surface_query = sim_utils.GLOBAL_CREATER.VectorField(n=3, dtype=float, shape=(self._batch_size, count))
        self._point_on_surface_answer = sim_utils.GLOBAL_CREATER.VectorField(n=4, dtype=float, shape=(self._batch_size, count))
        self._point_on_surface_query_n = sim_utils.GLOBAL_CREATER.ScalarField(dtype=int, shape=(self._batch_size, ))

        import open3d as o3d
        xyz_full = self._mesh.sample(int(count * fps_multiplier))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz_full)
        pcd_down = pcd.farthest_point_down_sample(count)
        xyz_down = np.asarray(pcd_down.points)

        self._point_on_surface_n = count
        self._point_on_surface.from_torch(torch.tensor(xyz_down, dtype=self._dtype, device=self._device))
        self._point_on_surface_query_n.fill(count)

    @ti.func
    def _query_self_surface_point_on_other_rigid_func(self, rigid: ti.template()):
        for batch_idx in range(self._batch_size):
            self._fwd[batch_idx] = so3.pos7d_to_matrix_func(self._pos[batch_idx])
        for batch_idx, pid in self._point_on_surface_query:
            self._point_on_surface_query[batch_idx, pid] = (
                self._fwd[batch_idx] @ 
                maths.vec3_pad1_func(self._scale[batch_idx] * self._point_on_surface[pid])
            )[:3]
        rigid._query_sdf_func(self._point_on_surface_query, self._point_on_surface_answer, self._point_on_surface_query_n)

    @ti.kernel
    def _check_self_surface_penetrate_other_rigid_kernel(self, 
                                                         rigid: ti.template(), 
                                                         sdf_th: float,
                                                         is_penetrate: ti.types.ndarray(dtype=int)):
        self._query_self_surface_point_on_other_rigid_func(rigid)
        for batch_idx in range(self._batch_size):
            is_penetrate[batch_idx] = 0
        for batch_idx, pid in self._point_on_surface_answer:
            s = self._point_on_surface_answer[batch_idx, pid][0]
            if s < sdf_th:
                is_penetrate[batch_idx] = 1

    def check_self_surface_penetrate_other_rigid(self, rigid: "Rigid", sdf_th: float) -> torch.Tensor:
        """int, [B, ], this function is slower because it uses ti.template() and compiles lot of times."""
        assert self._point_on_surface_n > 0, "No sample on surface."
        assert rigid._sdf is not None, f"Sdf in rigid:{rigid._name} not calculated."
        is_penetrate = torch.zeros((self._batch_size, ), dtype=self._dtype_int, device=self._device)
        self._check_self_surface_penetrate_other_rigid_kernel(rigid, sdf_th, is_penetrate)
        return is_penetrate
    
    @ti.kernel
    def _check_other_surface_penetrate_self_rigid_kernel(self, 
                                                         other_point_on_surface: ti.types.ndarray(dtype=ti.math.vec3), 
                                                         other_pos: ti.types.ndarray(dtype=sim_utils.vec7), 
                                                         other_scale: ti.types.ndarray(dtype=float), 
                                                         sdf_th: float,
                                                         is_penetrate: ti.types.ndarray(dtype=int)):
        for batch_idx, pid in self._point_on_surface_query:
            self._point_on_surface_query[batch_idx, pid] = (
                so3.pos7d_to_matrix_func(other_pos[batch_idx]) @ 
                maths.vec3_pad1_func(other_scale[batch_idx] * other_point_on_surface[pid])
            )[:3]
            self._query_sdf_func(self._point_on_surface_query, self._point_on_surface_answer, self._point_on_surface_query_n)
        for batch_idx in range(self._batch_size):
            is_penetrate[batch_idx] = 0
        for batch_idx, pid in self._point_on_surface_answer:
            s = self._point_on_surface_answer[batch_idx, pid][0]
            if s < sdf_th:
                is_penetrate[batch_idx] = 1

    def check_other_surface_penetrate_self_rigid(self, rigid: "Rigid", sdf_th: float) -> torch.Tensor:
        """int, [B, ]"""
        assert rigid._point_on_surface_n > 0, "No sample on surface."
        assert self._sdf is not None, f"Sdf in self:{self._name} not calculated."
        assert self._point_on_surface_query.shape[1] == rigid._point_on_surface.shape[0], \
            f"{self._point_on_surface_query.shape} {rigid._point_on_surface.shape}"
        is_penetrate = torch.zeros((self._batch_size, ), dtype=self._dtype_int, device=self._device)
        self._check_other_surface_penetrate_self_rigid_kernel(
            rigid._point_on_surface.to_torch(self.device), 
            rigid._pos.to_torch(self.device), 
            rigid._scale.to_torch(self.device), 
            sdf_th, is_penetrate
        )
        return is_penetrate

    @property
    def name(self) -> str:
        return self._name
        
    def get_mesh(self, batch_idx: int, **kwargs) -> trimesh.Trimesh:
        raise NotImplementedError
    
    def set_mesh(self, mesh: trimesh.Trimesh, sdf_mode: Literal["recalculate", "reinit", "none"], **kwargs):
        self._mesh = trimesh.Trimesh(
            vertices=mesh.vertices,
            faces=mesh.faces,
        )
        if sdf_mode == "recalculate":
            self._sdf.recalculate_sdf(torch.Tensor(mesh.vertices), torch.Tensor(mesh.faces))
        elif sdf_mode == "reinit":
            self._reinit_sdf(kwargs["sdf_cfg"], kwargs["global_cfg"])
        elif sdf_mode == "none":
            pass
        else:
            raise ValueError(sdf_mode)

    
    def get_mesh_shallow_raw(self) -> trimesh.Trimesh:
        return self._mesh
    
    def get_state(self) -> dict:
        state = {}
        for s in self._torch_state_str:
            state[s] = sim_utils.torch_to_numpy(getattr(self, f"_{s}"))
        for s in self._field_state_str:
            state[s] = getattr(self, f"_{s}").to_numpy()
        return state

    def set_state(self, state: dict):
        assert isinstance(state, dict)
        for s in self._torch_state_str:
            getattr(self, f"_{s}")[...] = torch.tensor(state[s])
        for s in self._field_state_str:
            getattr(self, f"_{s}").from_numpy(state[s])
    
    def reset(self):
        self._inertia[...] = torch.diag(torch.tensor([self._default_inertia] * 3))[None, ...]
        self._mass[...] = self._default_mass
        self._inertial_origin[...] = torch.eye(4)[None, ...]

        self._pos.fill(0.)
        self._vel.fill(0.)
        self._scale.fill(1.)


@ti.data_oriented
class RigidMesh(Rigid):
    """Rigid Mesh"""
    def __init__(self, rigid_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, mesh: trimesh.Trimesh) -> None:
        super().__init__(rigid_cfg, global_cfg)
        self._mesh = trimesh.Trimesh(vertices=mesh.vertices, faces=mesh.faces)
        if rigid_cfg.sdf_cfg.calculate_sdf:
            self._sdf = sdf.Sdf(self._mesh, rigid_cfg.sdf_cfg, global_cfg, self._name)
        if hasattr(rigid_cfg, "surface_sample"):
            self._sample_on_surface(rigid_cfg.surface_sample)

    def _get_mesh(self, batch_idx: int) -> trimesh.Trimesh:
        transform = (tra.translation_matrix(self._pos.to_numpy()[batch_idx, :3]) @ 
                     tra.quaternion_matrix(self._pos.to_numpy()[batch_idx, 3:]) @
                     tra.scale_matrix(self._scale.to_numpy()[batch_idx]))
        mesh = trimesh.Trimesh(vertices=self._mesh.vertices, faces=self._mesh.faces)
        mesh.apply_transform(transform)
        return mesh
    
    def get_mesh(self, batch_idx: int, **kwargs):
        return self._get_mesh(batch_idx)


@ti.data_oriented
class RigidBox(Rigid):
    """Rigid Box"""
    def __init__(self, rigid_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(rigid_cfg, global_cfg)

        self._extents: torch.Tensor = torch.zeros((3, ), dtype=self._dtype, device=self._device)
        """[3]"""
        self._extents[...] = torch.tensor(rigid_cfg.extents)
        self._mesh = self._get_mesh(0)
        if rigid_cfg.sdf_cfg.calculate_sdf:
            self._sdf = sdf.Sdf(self._mesh, rigid_cfg.sdf_cfg, global_cfg, self._name)
        if hasattr(rigid_cfg, "surface_sample"):
            self._sample_on_surface(rigid_cfg.surface_sample)

    def _get_mesh(self, batch_idx: int) -> trimesh.Trimesh:
        extents = sim_utils.torch_to_numpy(self._extents)
        transform = (tra.translation_matrix(self._pos.to_numpy()[batch_idx, :3]) @ 
                     tra.quaternion_matrix(self._pos.to_numpy()[batch_idx, 3:]) @
                     tra.scale_matrix(self._scale.to_numpy()[batch_idx]))
        mesh = trimesh.primitives.Box(extents=extents, transform=transform)
        return mesh
    
    def get_mesh(self, batch_idx: int, **kwargs):
        return self._get_mesh(batch_idx)
    
    def get_state(self) -> dict:
        state = super().get_state()
        state["extents"] = sim_utils.torch_to_numpy(self._extents)
        return state
    
    def set_state(self, state: dict):
        super().set_state(state)
        self._extents[...] = torch.tensor(state["extents"], dtype=self._dtype, device=self._device)


@ti.data_oriented
class RigidCylinder(Rigid):
    """Rigid Cylinder"""
    def __init__(self, rigid_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        super().__init__(rigid_cfg, global_cfg)

        self._radius: torch.Tensor = torch.zeros((2, ), dtype=self._dtype, device=self._device)
        self._radius[...] = torch.tensor(rigid_cfg.radius)

        self._height: torch.Tensor = torch.zeros((1, ), dtype=self._dtype, device=self._device)
        self._height[...] = torch.tensor(rigid_cfg.height)

        self._collision_facets = rigid_cfg.get("collision_facets", 16)
        self._visual_facets = rigid_cfg.get("visual_facets", 32)

        self._mesh = self._get_mesh(0, use_collision_instead_of_visual=True)
        if rigid_cfg.sdf_cfg.calculate_sdf:
            self._sdf = sdf.Sdf(self._mesh, rigid_cfg.sdf_cfg, global_cfg, self._name)
        if hasattr(rigid_cfg, "surface_sample"):
            self._sample_on_surface(rigid_cfg.surface_sample)

    def _get_mesh(self, batch_idx: int, use_collision_instead_of_visual: bool=False) -> trimesh.Trimesh:
        x, y = sim_utils.torch_to_numpy(self._radius)
        z, = sim_utils.torch_to_numpy(self._height)
        transform = (tra.translation_matrix(self._pos.to_numpy()[batch_idx, :3]) @ 
                     tra.quaternion_matrix(self._pos.to_numpy()[batch_idx, 3:]) @
                     tra.scale_matrix(self._scale.to_numpy()[batch_idx]) @
                     np.diag([x, y, z, 1.]).astype(z.dtype))
        mesh = trimesh.primitives.Cylinder(transform=transform,
                                           sections=self._collision_facets if use_collision_instead_of_visual else 
                                                    self._visual_facets)
        return mesh
    
    def get_height(self):
        """[1, ]"""
        return self._height.clone()
    
    def get_radius(self):
        """[2, ]"""
        return self._radius.clone()
    
    def get_mesh(self, batch_idx: int, use_collision_instead_of_visual: bool=False, **kwargs):
        return self._get_mesh(batch_idx, use_collision_instead_of_visual)

    def get_state(self) -> dict:
        state = super().get_state()
        state["radius"] = sim_utils.torch_to_numpy(self._radius)
        state["height"] = sim_utils.torch_to_numpy(self._height)
        return state
    
    def set_state(self, state: dict):
        super().set_state(state)
        self._radius[...] = torch.tensor(state["radius"], dtype=self._dtype, device=self._device)
        self._height[...] = torch.tensor(state["height"], dtype=self._dtype, device=self._device)
