import taichi as ti

from typing import Dict, Union
import copy

import torch
import trimesh

import omegaconf

from .sim_utils import BaseClass
from .rigid import Rigid
from .articulate import Articulate
from .cloth import Cloth
from .collision_force import ClothForceCollision
from .collision_position import ClothPositionCollision
from .sim_cloth import ClothSim
from .sim_rigid import RigidSim
from .sim_articulate import ArticulateSim
from .spatial import SpatialPartition
from . import sim_utils


@ti.data_oriented
class Sim(BaseClass):
    def __init__(self, sim_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        self._sim_cfg = copy.deepcopy(sim_cfg)
        self._global_cfg = copy.deepcopy(global_cfg)

        super().__init__(global_cfg)

        self._step_dt: float = self._sim_cfg.step_dt
        self._substeps: int = self._sim_cfg.substeps
        self._dt = self._step_dt / self._substeps

        self._rigid_map: Dict[str, Rigid] = {}
        self._articulate_map: Dict[str, Articulate] = {}
        self._cloth_map: Dict[str, Cloth] = {}

        # cloth simulation
        self._cloth_simulation_map: Dict[str, Union[ClothSim, None]] = {}

        # rigid simulation
        self._rigid_simulation_map: Dict[str, Union[RigidSim, None]] = {}

        # articulate simulation
        self._articulate_simulation_map: Dict[str, Union[ArticulateSim, None]] = {}

        # collisions
        self._cloth_force_collision_map: Dict[str, Dict[str, ClothForceCollision]] = {}
        """cloth_name -> collision_name -> collision_obj"""
        self._cloth_position_collision_map: Dict[str, Dict[str, ClothPositionCollision]] = {}
        """cloth_name -> collision_name -> collision_obj"""

        # spatial data structure
        if hasattr(self._sim_cfg, "spatial_cfg"):
            self._spatial = SpatialPartition(self._sim_cfg.spatial_cfg, self._global_cfg)

        # gravity
        self._gravity: torch.Tensor = torch.zeros((self._batch_size, 3), dtype=self._dtype, device=self._device)
        """[B, 3]"""
        self._gravity[...] = torch.tensor([self._sim_cfg.get("gravity", (0., 0., -9.8))] * self._batch_size)

        # global_fatal_flag:
        self._global_fatal_flag: torch.Tensor = torch.zeros((self._batch_size, ), dtype=self._dtype_int, device=self._device)
        """int, [B, ]"""

    @property
    def rigid_map(self) -> Dict[str, Rigid]:
        return self._rigid_map
    
    @property
    def articulate_map(self) -> Dict[str, Articulate]:
        return self._articulate_map
    
    @property
    def cloth_map(self) -> Dict[str, Cloth]:
        return self._cloth_map
    
    @property
    def step_dt(self) -> float:
        return self._step_dt

    @property
    def substeps(self) -> int:
        return self._substeps
    
    @property
    def dt(self) -> float:
        return self._dt
    
    def set_gravity(self, gravity: torch.Tensor):
        self._gravity[...] = gravity

    def set_num_substeps(self, new_num_substeps: int) -> float:
        self._substeps = int(new_num_substeps)
        self._dt = self._step_dt / self._substeps

    def _add_rigid(self, rigid: Rigid):
        assert isinstance(rigid, Rigid)
        assert rigid._name not in self._rigid_map.keys()
        self._rigid_map[rigid._name] = rigid
        self._rigid_simulation_map[rigid._name] = None

    def _add_articulate(self, articulate: Articulate):
        assert isinstance(articulate, Articulate)
        assert articulate._name not in self._articulate_map.keys()
        self._articulate_map[articulate._name] = articulate
        self._articulate_simulation_map[articulate._name] = None

    def _add_cloth(self, cloth: Cloth):
        assert isinstance(cloth, Cloth)
        assert cloth._name not in self._cloth_map.keys()
        self._cloth_map[cloth._name] = cloth
        self._cloth_simulation_map[cloth._name] = None
        self._cloth_force_collision_map[cloth._name] = {}
        self._cloth_position_collision_map[cloth._name] = {}

    def _add_cloth_force_collision(self, cloth_collision: ClothForceCollision):
        assert isinstance(cloth_collision, ClothForceCollision)
        cloth_name = cloth_collision._cloth._name

        assert cloth_name in self._cloth_map.keys()
        self._cloth_simulation_map[cloth_name] = None # need to re-init ClothSim
        
        assert cloth_collision._name not in self._cloth_force_collision_map[cloth_name].keys()
        self._cloth_force_collision_map[cloth_name][cloth_collision._name] = cloth_collision # add collision

    def _add_cloth_position_collision(self, cloth_collision: ClothPositionCollision):
        assert isinstance(cloth_collision, ClothPositionCollision)
        cloth_name = cloth_collision._cloth._name
        
        assert cloth_collision._name not in self._cloth_position_collision_map[cloth_name].keys()
        self._cloth_position_collision_map[cloth_name][cloth_collision._name] = cloth_collision # add collision

    def _step_vel_cloth_impl(self, cloth: Cloth):
        cloth_name = cloth._name
        if self._cloth_simulation_map[cloth_name] is None:
            assert hasattr(self._sim_cfg, cloth_name), f"missing {cloth_name} in sim_cfg"
            # initialize ClothSim
            self._cloth_simulation_map[cloth_name] = ClothSim(
                cloth,
                list(self._cloth_force_collision_map[cloth_name].values()),
                list(self._cloth_position_collision_map[cloth_name].values()),
                getattr(self._sim_cfg, cloth_name),
                self._global_cfg
            )
        self._cloth_simulation_map[cloth_name]._step_vel_cloth(self._dt, self._spatial, self._global_fatal_flag, self._gravity)

    def _step_vel_rigid_impl(self, rigid: Rigid):
        rigid_name = rigid._name
        if self._rigid_simulation_map[rigid_name] is None:
            assert hasattr(self._sim_cfg, rigid_name), f"missing {rigid_name} in sim_cfg"
            # initialize RigidSim
            self._rigid_simulation_map[rigid_name] = RigidSim(
                rigid,
                getattr(self._sim_cfg, rigid_name),
                self._global_cfg,
            )
        self._rigid_simulation_map[rigid_name]._step_vel_rigid(self._dt, self._gravity)

    def _step_vel_articulate_impl(self, articulate: Articulate):
        articulate_name = articulate._name
        if self._articulate_simulation_map[articulate_name] is None:
            assert hasattr(self._sim_cfg, articulate_name), f"missing {articulate_name} in sim_cfg"
            # initialize RigidSim
            self._articulate_simulation_map[articulate_name] = ArticulateSim(
                articulate,
                getattr(self._sim_cfg, articulate_name),
                self._global_cfg,
            )
        self._articulate_simulation_map[articulate_name]._step_vel_articulate(self._dt, self._gravity)

    def _step_vel(self, obj: Union[Cloth, Rigid, Articulate]):
        if isinstance(obj, Cloth):
            self._step_vel_cloth_impl(obj)
        elif isinstance(obj, Rigid):
            self._step_vel_rigid_impl(obj)
        elif isinstance(obj, Articulate):
            self._step_vel_articulate_impl(obj)
        else:
            raise NotImplementedError(type(obj))
        
    def _step_pos_cloth_impl(self, cloth: Cloth):
        cloth_name = cloth._name
        self._cloth_simulation_map[cloth_name]._step_pos_cloth(self._dt, self._spatial, self._global_fatal_flag)

    def _step_pos_rigid_impl(self, rigid: Rigid):
        rigid_name = rigid._name
        self._rigid_simulation_map[rigid_name]._step_pos_rigid(self._dt)

    def _step_pos_articulate_impl(self, articulate: Articulate):
        articulate_name = articulate._name
        self._articulate_simulation_map[articulate_name]._step_pos_articulate(self._dt)
        
    def _step_pos(self, obj: Union[Cloth, Rigid, Articulate]):
        if isinstance(obj, Cloth):
            self._step_pos_cloth_impl(obj)
        elif isinstance(obj, Rigid):
            self._step_pos_rigid_impl(obj)
        elif isinstance(obj, Articulate):
            self._step_pos_articulate_impl(obj)
        else:
            raise NotImplementedError(type(obj))

    def get_global_fatal_flag(self):
        """int, [B, ]"""
        return self._global_fatal_flag.clone()
    
    def _detect_cloth_self_intersection(self, cloth_name: str):
        return self._cloth_simulation_map[cloth_name]._detect_self_intersection(self._spatial)
    
    def detect_cloth_self_intersection(self, cloth_name: str):
        """return int, [B, ]"""
        return self._detect_cloth_self_intersection(cloth_name)

    def get_mesh(self, batch_idx: int, **kwargs) -> trimesh.Trimesh:
        meshes = []
        for a in self._rigid_map.values():
            meshes.append(a.get_mesh(batch_idx, **kwargs))
        for a in self._articulate_map.values():
            meshes.append(a.get_mesh(batch_idx, **kwargs))
        for a in self._cloth_map.values():
            meshes.append(a.get_mesh(batch_idx, **kwargs))
        return trimesh.util.concatenate(meshes)

    def get_state(self) -> dict:
        state = {
            "rigid": {
                k: v.get_state() for k, v in self._rigid_map.items()
            },
            "articulate": {
                k: v.get_state() for k, v in self._articulate_map.items()
            },
            "cloth": {
                k: v.get_state() for k, v in self._cloth_map.items()
            },
            "global_fatal_flag": sim_utils.torch_to_numpy(self._global_fatal_flag)
        }
        return state
    
    def set_state(self, state: dict):
        assert isinstance(state, dict)

        for k, v in state["rigid"].items():
            self._rigid_map[k].set_state(v)
        for k, v in state["articulate"].items():
            self._articulate_map[k].set_state(v)
        for k, v in state["cloth"].items():
            self._cloth_map[k].set_state(v)
        self._global_fatal_flag[...] = torch.tensor(state["global_fatal_flag"])

    def reset(self):
        """Reset rigid, articulate objects."""
        for v in self._rigid_map.values():
            v.reset()
        for v in self._articulate_map.values():
            v.reset()
        for v in self._cloth_map.values():
            v.reset()
        self._global_fatal_flag[...] = 0