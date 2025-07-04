import os
import copy

from typing import Dict, Union, Literal, Union, List, Iterable, Optional, Callable, Any
import datetime

import torch
import numpy as np
import trimesh
import json

import hydra
import omegaconf


def torch_to_numpy(t: torch.Tensor) -> np.ndarray:
    return t.detach().cpu().numpy().copy()


def torch_dict_clone(d: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {k: v.clone() for k, v in d.items()}


def torch_dict_to_numpy_dict(d: Dict[str, Union[torch.Tensor, int, float, dict]]) -> Dict[str, Union[np.ndarray, int, float, dict]]:
    ret = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            ret[k] = torch_to_numpy(v)
        elif isinstance(v, dict):
            ret[k] = torch_dict_to_numpy_dict(v)
        elif isinstance(v, (float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def torch_dict_to_list_dict(d: Dict[str, Union[torch.Tensor, int, float, dict]]) -> Dict[str, Union[list, int, float, dict]]:
    ret = {}
    for k, v in d.items():
        if isinstance(v, torch.Tensor):
            ret[k] = torch_to_numpy(v).tolist()
        elif isinstance(v, dict):
            ret[k] = torch_dict_to_list_dict(v)
        elif isinstance(v, (float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def extract_single_batch(d: Dict[str, Union[np.ndarray, dict, str, float, int]], batch_idx):
    """each array in ret is (1, ...)"""
    ret = {}
    for k, v in d.items():
        if isinstance(v, np.ndarray):
            ret[k] = v[[batch_idx], ...]
        elif isinstance(v, dict):
            ret[k] = extract_single_batch(v, batch_idx)
        elif isinstance(v, (str, float, int)):
            ret[k] = v
        else:
            raise TypeError(type(v))
    return ret


def merge_single_batch(d1: Dict[str, Union[np.ndarray, dict, str, float, int]], d2: Dict[str, Union[np.ndarray, dict, str, float, int]]):
    ret = {}
    for k, v1 in d1.items():
        v2 = d2[k]
        assert isinstance(v1, type(v2)), f"{type(v1)} {type(v2)}"
        if isinstance(v1, np.ndarray):
            ret[k] = np.concatenate([v1, v2], axis=0)
        elif isinstance(v1, dict):
            ret[k] = merge_single_batch(v1, v2)
        elif isinstance(v1, (str, float, int)):
            ret[k] = v1
        else:
            raise TypeError(type(v1))
    return ret


def get_folder_size(folder):
    total_size = 0
    for dirpath, dirnames, filenames in os.walk(folder):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            try:
                size = os.path.getsize(fp)
                total_size += size
            except OSError:
                pass  # Ignore files that can't be accessed
    return total_size


def list_or(*args):
    ret = None
    for i, x in enumerate(args):
        assert isinstance(x, list), f"args:{args} x:{x}"
        if i == 0:
            ret = np.array(x, dtype=bool)
        else:
            ret |= np.array(x, dtype=bool)
    return ret.tolist()


def list_and(*args):
    ret = None
    for i, x in enumerate(args):
        assert isinstance(x, list), f"args:{args} x:{x}"
        if i == 0:
            ret = np.array(x, dtype=bool)
        else:
            ret &= np.array(x, dtype=bool)
    return ret.tolist()


def index_list(l: Union[list, dict]) -> dict:
    if isinstance(l, (list, dict)):
        ret = {}
        for k, v in l.items() if isinstance(l, dict) else enumerate(l):
            if isinstance(v, (list, dict)):
                ret[k] = index_list(v)
            else:
                ret[k] = v
        return ret
    else:
        raise TypeError(f"{type(l)}, {l}")
    

def map_01_ab(x, a, b):
    """`a + x * (b - a)` map a uniform distribution x in [0, 1] to [a, b]"""
    return a + x * (b - a)


def format_int(x: int, x_max: int):
    """
    Example:
    ```
    N = 100
    for x in range(N):
        print(format_int(x, N-1)) # 00, ..., 99
    ```
    """
    return str(x).zfill(len(str(x_max)))
    

_path_handler = hydra.utils.to_absolute_path


def set_path_handler(path_handler: Callable):
    assert callable(path_handler)
    global _path_handler
    _path_handler = path_handler


def default_path_handler():
    return hydra.utils.to_absolute_path


def get_path_handler():
    """to_absolute_path"""
    global _path_handler
    return _path_handler


def omegaconf_resolve_recursive(cfg: Union[omegaconf.DictConfig, omegaconf.ListConfig]):
    while True:
        old_cfg = copy.deepcopy(cfg)
        omegaconf.OmegaConf.resolve(cfg)
        if old_cfg == cfg:
            break


def init_omegaconf():
    def func_load(s, v=""):
        d = omegaconf.OmegaConf.load(_path_handler(str(s)))
        x = eval(f"d{str(v)}")
        return x
    
    def func_mean(s):
        return sum(s) / len(s)
    
    def func_eval(s):
        result = eval(str(s))
        print(f"eval [{str(s)}] ... result is [{type(result)} {result}]")
        return result
    
    omegaconf.OmegaConf.clear_resolvers()
    omegaconf.OmegaConf.register_new_resolver("_load_", func_load)
    omegaconf.OmegaConf.register_new_resolver("_mean_", func_mean)
    omegaconf.OmegaConf.register_new_resolver("_eval_", func_eval)
    if not omegaconf.OmegaConf.has_resolver("now"):
        omegaconf.OmegaConf.register_new_resolver(
            "now",
            lambda pattern: datetime.datetime.now().strftime(pattern),
            use_cache=True,
            replace=True,
        )


def resolve_overwrite(cfg):
    overwrite_cfg = getattr(cfg, "overwrite", omegaconf.DictConfig({}))
    if overwrite_cfg is None:
        overwrite_cfg = omegaconf.DictConfig({})
    return omegaconf.OmegaConf.merge(cfg, overwrite_cfg)


def ddp_is_rank_0() -> bool:
    return int(os.environ.get('LOCAL_RANK', 0)) == 0 and int(os.environ.get('NODE_RANK', 0)) == 0