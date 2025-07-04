import logging
logger = logging.getLogger(__name__)

import os
import sys

import pathlib
import matplotlib
matplotlib.use('Agg')

import hydra
import hydra.core.utils as hydra_utils
import omegaconf

import numpy as np
import torch

import robohang.common.utils as utils
from robohang.real.realapi import RealAPI
from robohang.real.world import RealWorld
import robohang.sim.api as api

def overwrite_hanger(cfg: omegaconf.DictConfig, hanger_idx: int):
    cfg.world.obj.hanger.mesh_path = f"assets/hanger/{hanger_idx}/hanger.obj"
    if hanger_idx % 2 == 1: # hanger without rack
        cfg.world.primitive.insert.rotate.rotate.h = 0.05


@hydra.main(config_path="../config/real/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    print(f"pid:{os.getpid()}")
    logger.info(f"pid:{os.getpid()}")
    logger.info(" ".join(sys.argv))

    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    overwrite_hanger(cfg, cfg.hanger_idx)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))

    # init taichi
    api.init_taichi(cfg.setup.taichi, cfg.glb_cfg)

    real_api = RealAPI(cfg.realapi)
    world = RealWorld(real_api, cfg.world, getattr(cfg.policy, cfg.policy.type), cfg.glb_cfg)

    hydra_cfg = omegaconf.OmegaConf.load(".hydra/hydra.yaml")
    hydra_utils.configure_log(hydra_cfg.hydra.job_logging, hydra_cfg.hydra.verbose)
    world.run_human_control()
        
if __name__ == "__main__":
    main()
