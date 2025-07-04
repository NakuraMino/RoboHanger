import logging
logger = logging.getLogger(__name__)

import taichi as ti

import os
import sys
import pathlib
import pickle
import copy
import json
from typing import Dict, Optional
import pprint

import numpy as np
import torch

import trimesh
import trimesh.transformations as tra
import tqdm

import omegaconf
import hydra

import robohang.common.utils as utils
import robohang.sim.api as api
from robohang.sim.sim import Sim
from robohang.sim.env import Env
import robohang.sim.sim_utils as sim_utils
from robohang.env.sim_env import SimEnv
from robohang.agent.base_agent import BaseAgent, GalbotZeroAgent, GalbotOneAgent
from robohang.policy.funnel.funnel_gym import FunnelGym
from robohang.policy.funnel.funnel_policy import FunnelPolicyUNet
from robohang.policy.policy_utils import ObservationExporter


@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    print(f"pid:{os.getpid()}")
    logger.info(f"pid:{os.getpid()}")
    logger.info(" ".join(sys.argv))

    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))
    callbacks = []
    
    # setup device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.setup.cuda)

    # init taichi
    api.init_taichi(cfg.setup.taichi, cfg.glb_cfg)

    # init sim env
    gym = api.acquire_gym()
    sim = gym.create_sim(cfg.sim, cfg.glb_cfg)
    env = gym.create_env(sim, cfg.glb_cfg)

    # constants
    batch_size = int(cfg.glb_cfg.batch_size)
    dtype, dtype_int, device = sim.dtype, sim.dtype_int, sim.device

    # asset
    table_path = utils.get_path_handler()(cfg.asset.table.mesh_path)
    table_mesh = trimesh.load_mesh(table_path)
    table = gym.create_rigid(sim, cfg.asset.table.cfg, cfg.glb_cfg, mesh=table_mesh)
    table.set_pos(torch.tensor([cfg.asset.table.pos] * batch_size))

    garment_path = utils.get_path_handler()(cfg.asset.garment.mesh_path)
    garment_mesh = trimesh.load_mesh(garment_path)
    garment = gym.create_cloth(sim, garment_mesh, cfg.asset.garment.cfg, cfg.glb_cfg)
    garment.set_pos(garment.get_pos() + torch.tensor(cfg.asset.garment.translation, dtype=dtype, device=device))
    garment_meta = omegaconf.OmegaConf.load(garment_path + ".meta.yaml")

    # add collision
    garment_self = gym.add_cloth_self_force_collision(
        sim, garment, cfg.sim.garment_self_collision, cfg.glb_cfg
    )
    garment_table = gym.add_cloth_rigid_position_collision(
        sim, garment, table, cfg.sim.garment_table, cfg.glb_cfg
    )

    # callback
    class Exporter:
        def __init__(self) -> None:
            self.current_cnt = 0
        def callback(self, env: Env, sim: Sim, substep: int ):
            if substep == sim.substeps - 1:
                gm = garment.get_mesh(0)
                trimesh.util.concatenate([
                    trimesh.Trimesh(vertices=gm.vertices, faces=gm.faces, vertex_colors=garment_mesh.visual.vertex_colors)
                ]).export(f"{str(self.current_cnt).zfill(6)}.obj")
                self.current_cnt += 1
    exp = Exporter()
    callbacks = [exp.callback]
    
    for i in tqdm.tqdm(range(cfg.output.steps), dynamic_ncols=True):
        gym.simulate(env, sim, callbacks)
    

if __name__ == "__main__":
    main()