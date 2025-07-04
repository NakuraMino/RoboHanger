import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import Callable, List

import trimesh

import omegaconf

from .env import Env
from .sim import Sim
from .rigid import Rigid, RigidBox, RigidMesh, RigidCylinder
from .articulate import Articulate
from .actor import ArticulateActor, RigidActor
from .cloth import Cloth
from .collision_force import ClothSelfForceCollision, ClothRigidForceCollision, ClothArticulateForceCollision
from .collision_position import ClothRigidPositionCollision, ClothArticulatePositionCollision
from . import sim_utils

class Gym:
    def __init__(self) -> None:
        self._total_substep = 0
        self._total_step = 0

    @property
    def total_substep(self):
        return self._total_substep
    
    @property
    def total_step(self):
        return self._total_step

    def create_sim(self, sim_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> Sim:
        assert isinstance(sim_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        return Sim(sim_cfg, global_cfg)
    
    def create_env(self, sim: Sim, global_cfg: omegaconf.DictConfig) -> Env:
        assert isinstance(sim, Sim)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        env = Env(sim, global_cfg)
        return env
    
    def create_rigid(self, sim: Sim, rigid_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig, **kwargs) -> Rigid:
        assert isinstance(sim, Sim)
        assert isinstance(rigid_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in rigid_cfg.keys():
            rigid_cfg = omegaconf.OmegaConf.to_container(rigid_cfg)
            rigid_cfg["name"] = sim_utils.unique_name(rigid_cfg["type"], sim.rigid_map.keys())
            rigid_cfg = omegaconf.DictConfig(rigid_cfg)
        else:
            assert rigid_cfg.name not in sim.rigid_map.keys(), \
                f"name:{rigid_cfg.name} already in {list(sim.rigid_map.keys())}"

        if rigid_cfg.type == "box":
            rigid = RigidBox(rigid_cfg, global_cfg)
        elif rigid_cfg.type == "cylinder":
            rigid = RigidCylinder(rigid_cfg, global_cfg)
        elif rigid_cfg.type == "mesh":
            assert "mesh" in kwargs.keys(), "missing kwargs 'mesh' (type: trimesh.Trimesh) in gym.create_rigid()"
            rigid = RigidMesh(rigid_cfg, global_cfg, kwargs["mesh"])
        else:
            raise NotImplementedError(rigid_cfg.type)
        
        sim._add_rigid(rigid)
        return rigid
    
    def create_articulate(self, sim: Sim, articulate_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> Articulate:
        assert isinstance(sim, Sim)
        assert isinstance(articulate_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in articulate_cfg.keys():
            articulate_cfg = omegaconf.OmegaConf.to_container(articulate_cfg)
            articulate_cfg["name"] = sim_utils.unique_name("articulate", sim.articulate_map.keys())
            articulate_cfg = omegaconf.DictConfig(articulate_cfg)
        else:
            assert articulate_cfg.name not in sim.articulate_map.keys(), \
                f"name:{articulate_cfg.name} already in {list(sim.articulate_map.keys())}"

        articulate = Articulate(articulate_cfg, global_cfg)
        
        sim._add_articulate(articulate)
        return articulate

    def create_cloth(self, sim: Sim, mesh: trimesh.Trimesh, cloth_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> Cloth:
        assert isinstance(sim, Sim)
        assert isinstance(cloth_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in cloth_cfg.keys():
            cloth_cfg = omegaconf.OmegaConf.to_container(cloth_cfg)
            cloth_cfg["name"] = sim_utils.unique_name("cloth", sim.cloth_map.keys())
            cloth_cfg = omegaconf.DictConfig(cloth_cfg)
        else:
            assert cloth_cfg.name not in sim.cloth_map.keys(), \
                f"name:{cloth_cfg.name} already in {list(sim.cloth_map.keys())}"

        cloth = Cloth(mesh, cloth_cfg, global_cfg)
        sim._add_cloth(cloth)

        return cloth
    
    def create_articulate_actor(self, env: Env, articulate: Articulate, actor_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> ArticulateActor:
        assert isinstance(env, Env)
        assert isinstance(articulate, Articulate)
        assert isinstance(actor_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in actor_cfg.keys():
            actor_cfg = omegaconf.OmegaConf.to_container(actor_cfg)
            actor_cfg["name"] = sim_utils.unique_name("articulate_actor", env.actor_map.keys())
            actor_cfg = omegaconf.DictConfig(actor_cfg)
        else:
            assert actor_cfg.name not in env.actor_map.keys(), \
                f"name:{actor_cfg.name} already in {list(env.actor_map.keys())}"

        artor = ArticulateActor(articulate, actor_cfg, global_cfg)
        
        env._add_actor(artor)
        return artor
    
    def create_rigid_actor(self, env: Env, rigid: Rigid, actor_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> RigidActor:
        assert isinstance(env, Env)
        assert isinstance(rigid, Rigid)
        assert isinstance(actor_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in actor_cfg.keys():
            actor_cfg = omegaconf.OmegaConf.to_container(actor_cfg)
            actor_cfg["name"] = sim_utils.unique_name("rigid_actor", env.actor_map.keys())
            actor_cfg = omegaconf.DictConfig(actor_cfg)
        else:
            assert actor_cfg.name not in env.actor_map.keys(), \
                f"name:{actor_cfg.name} already in {list(env.actor_map.keys())}"

        artor = RigidActor(rigid, actor_cfg, global_cfg)
        
        env._add_actor(artor)
        return artor
    
    def add_cloth_self_force_collision(self, sim: Sim, cloth: Cloth, self_collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> ClothSelfForceCollision:
        assert isinstance(sim, Sim)
        assert isinstance(cloth, Cloth)
        assert isinstance(self_collision_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in self_collision_cfg.keys():
            self_collision_cfg = omegaconf.OmegaConf.to_container(self_collision_cfg)
            self_collision_cfg["name"] = sim_utils.unique_name(f"{cloth._name}_self_collision", sim._cloth_force_collision_map[cloth._name].keys())
            self_collision_cfg = omegaconf.DictConfig(self_collision_cfg)
        else:
            assert self_collision_cfg.name not in sim._cloth_force_collision_map[cloth._name].keys(), \
                f"name:{self_collision_cfg.name} already in {list(sim._cloth_force_collision_map[cloth._name].keys())}"

        self_collision = ClothSelfForceCollision(cloth, self_collision_cfg, global_cfg)
        sim._add_cloth_force_collision(self_collision)

        return self_collision
    
    def add_cloth_rigid_force_collision(self, sim: Sim, cloth: Cloth, rigid: Rigid, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        assert isinstance(sim, Sim)
        assert isinstance(cloth, Cloth)
        assert isinstance(rigid, Rigid)
        assert isinstance(collision_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in collision_cfg.keys():
            collision_cfg = omegaconf.OmegaConf.to_container(collision_cfg)
            collision_cfg["name"] = sim_utils.unique_name(f"{cloth._name}_{rigid._name}_force_collision", sim._cloth_force_collision_map[cloth._name].keys())
            collision_cfg = omegaconf.DictConfig(collision_cfg)
        else:
            assert collision_cfg.name not in sim._cloth_force_collision_map[cloth._name].keys(), \
                f"name:{collision_cfg.name} already in {list(sim._cloth_force_collision_map[cloth._name].keys())}"

        collision = ClothRigidForceCollision(cloth, rigid, collision_cfg, global_cfg)
        sim._add_cloth_force_collision(collision)
        return collision
    
    def add_cloth_rigid_position_collision(self, sim: Sim, cloth: Cloth, rigid: Rigid, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        assert isinstance(sim, Sim)
        assert isinstance(cloth, Cloth)
        assert isinstance(rigid, Rigid)
        assert isinstance(collision_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in collision_cfg.keys():
            collision_cfg = omegaconf.OmegaConf.to_container(collision_cfg)
            collision_cfg["name"] = sim_utils.unique_name(f"{cloth._name}_{rigid._name}_position_collision", sim._cloth_position_collision_map[cloth._name].keys())
            collision_cfg = omegaconf.DictConfig(collision_cfg)
        else:
            assert collision_cfg.name not in sim._cloth_position_collision_map[cloth._name].keys(), \
                f"name:{collision_cfg.name} already in {list(sim._cloth_position_collision_map[cloth._name].keys())}"

        collision = ClothRigidPositionCollision(cloth, rigid, collision_cfg, global_cfg)
        sim._add_cloth_position_collision(collision)
        return collision

    def add_cloth_articulate_force_collision(self, sim: Sim, cloth: Cloth, articulate: Articulate, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        assert isinstance(sim, Sim)
        assert isinstance(cloth, Cloth)
        assert isinstance(articulate, Articulate)
        assert isinstance(collision_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in collision_cfg.keys():
            collision_cfg = omegaconf.OmegaConf.to_container(collision_cfg)
            collision_cfg["name"] = sim_utils.unique_name(f"{cloth._name}_{articulate._name}_force_collision", sim._cloth_force_collision_map[cloth._name].keys())
            collision_cfg = omegaconf.DictConfig(collision_cfg)
        else:
            assert collision_cfg.name not in sim._cloth_force_collision_map[cloth._name].keys(), \
                f"name:{collision_cfg.name} already in {list(sim._cloth_force_collision_map[cloth._name].keys())}"

        collision = ClothArticulateForceCollision(cloth, articulate, collision_cfg, global_cfg)
        sim._add_cloth_force_collision(collision)
        return collision
    
    def add_cloth_articulate_position_collision(self, sim: Sim, cloth: Cloth, articulate: Articulate, collision_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
        assert isinstance(sim, Sim)
        assert isinstance(cloth, Cloth)
        assert isinstance(articulate, Articulate)
        assert isinstance(collision_cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        if "name" not in collision_cfg.keys():
            collision_cfg = omegaconf.OmegaConf.to_container(collision_cfg)
            collision_cfg["name"] = sim_utils.unique_name(f"{cloth._name}_{articulate._name}_position_collision", sim._cloth_position_collision_map[cloth._name].keys())
            collision_cfg = omegaconf.DictConfig(collision_cfg)
        else:
            assert collision_cfg.name not in sim._cloth_position_collision_map[cloth._name].keys(), \
                f"name:{collision_cfg.name} already in {list(sim._cloth_position_collision_map[cloth._name].keys())}"

        collision = ClothArticulatePositionCollision(cloth, articulate, collision_cfg, global_cfg)
        sim._add_cloth_position_collision(collision)
        return collision
    
    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_actor(self, env: Env, sim: Sim):
        for actor in env.actor_map.values():
            actor._step(sim.dt)

    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_rigid_vel(self, env: Env, sim: Sim):
        for rigid in sim.rigid_map.values():
            sim._step_vel(rigid)

    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_articulate_vel(self, env: Env, sim: Sim):
        for articulate in sim.articulate_map.values():
            sim._step_vel(articulate)

    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_cloth_vel(self, env: Env, sim: Sim):
        for cloth in sim.cloth_map.values():
            sim._step_vel(cloth)

    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_rigid_pos(self, env: Env, sim: Sim):
        for rigid in sim.rigid_map.values():
            sim._step_pos(rigid)

    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_articulate_pos(self, env: Env, sim: Sim):
        for articulate in sim.articulate_map.values():
            sim._step_pos(articulate)
    
    @sim_utils.GLOBAL_TIMER.timer
    def _gym_step_cloth_pos(self, env: Env, sim: Sim):
        for cloth in sim.cloth_map.values():
            sim._step_pos(cloth)

    @sim_utils.GLOBAL_TIMER.timer
    def _call_callbacks(self, env: Env, sim: Sim, callbacks: List[Callable[[Env, Sim, int], None]], substep: int):
        for callback in callbacks:
            if callable(callback):
                callback(env, sim, substep)
    
    def simulate(self, env: Env, sim: Sim, callbacks: List[Callable[[Env, Sim, int], None]]=None):
        assert isinstance(env, Env)
        assert isinstance(sim, Sim)
        if callbacks is None:
            callbacks = []
        assert hasattr(callbacks, "__iter__")

        for substep in range(sim._substeps):
            logger.info(f"current_total_substep:{self._total_substep} current_step:{self._total_step}")
            logger.info(f"global_fatal_flag: {sim._global_fatal_flag}")

            self._gym_step_actor(env, sim)

            self._gym_step_rigid_vel(env, sim)
            self._gym_step_articulate_vel(env, sim)
            self._gym_step_cloth_vel(env, sim)

            self._gym_step_rigid_pos(env, sim)
            self._gym_step_articulate_pos(env, sim)
            self._gym_step_cloth_pos(env, sim)
            
            logger.info(f"call_callbacks")
            self._call_callbacks(env, sim, callbacks, substep)
            self._total_substep += 1
        
        self._total_step += 1
