import taichi as ti

import random
import numpy as np
import torch

import omegaconf

from .gym import Gym

def init_taichi(taichi_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig):
    taichi_cfg = omegaconf.OmegaConf.to_container(taichi_cfg)
    taichi_cfg["arch"] = getattr(ti, taichi_cfg["arch"])

    if global_cfg.default_float == "float64":
        taichi_cfg["default_fp"] = ti.f64
    elif global_cfg.default_float == "float32":
        taichi_cfg["default_fp"] = ti.f32
    else:
        raise NotImplementedError(global_cfg.default_float)
    if global_cfg.default_int == "int64":
        taichi_cfg["default_ip"] = ti.i64
    elif global_cfg.default_int == "int32":
        taichi_cfg["default_ip"] = ti.i32
    else:
        raise NotImplementedError(global_cfg.default_int)
    
    if hasattr(global_cfg, "seed"):
        seed = global_cfg.seed
        taichi_cfg["random_seed"] = seed
        np.random.seed(seed)
        torch.random.manual_seed(seed)
        random.seed(seed, version=2)
    
    ti.init(**taichi_cfg)

def acquire_gym() -> Gym:
    return Gym()
