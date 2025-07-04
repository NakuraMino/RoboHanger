import taichi as ti

import inspect
import os
import sys
import logging
import time
from typing import Iterable, Dict, List, Callable, Tuple, Union
from dataclasses import dataclass

import torch
import numpy as np

from hydra.utils import to_absolute_path

import trimesh
import omegaconf

from ..common.utils import torch_dict_clone, torch_dict_to_numpy_dict, torch_to_numpy


@ti.kernel
def return_true_kernel() -> bool:
    return True


MAX_RANGE = 2e9
vec6 = ti.types.vector(6, float)
vec7 = ti.types.vector(7, float)

class TimerFuncName:
    def __init__(self, cls:str, name: str, filepath: str, lineno: int) -> None:
        self._cls = str(cls)
        self._name = str(name)
        self._filepath = str(filepath)
        self._lineno = str(lineno)
        self._val = (cls, name, filepath, lineno)

    def __hash__(self) -> int:
        return self._val.__hash__()
    
    def __eq__(self, __value: object) -> bool:
        assert isinstance(__value, TimerFuncName)
        return self._val == __value._val
    
    def __lt__(self, other) -> bool:
        assert isinstance(other, TimerFuncName)
        if self._cls != other._cls:
            return self._cls < other._cls
        elif self._filepath != other._filepath:
            return self._filepath < other._filepath
        else:
            return self._lineno < other._lineno
        
    def to_str(self, start_path=None) -> str:
        return f"{self._cls}: {self._name} at file '{os.path.relpath(self._filepath, start_path)}', line {self._lineno}"


class Timer:
    def __init__(self) -> None:
        self._result: Dict[TimerFuncName, List[float]] = {}
        self._enable: bool = False

    def timer(self, func: Callable):
        def wrapper(*args, **kwargs):
            if self._enable:
                return_true_kernel() # wait all the kernel to finish
                tic = time.time()
                result = func(*args, **kwargs)
                return_true_kernel() # wait all the kernel to finish
                toc = time.time()
                args_name = inspect.getfullargspec(func)[0]
                name = TimerFuncName(
                    args[0].__class__.__name__ if args_name and args_name[0] == "self" else "",
                    func.__name__, 
                    to_absolute_path(inspect.currentframe().f_back.f_code.co_filename),
                    inspect.currentframe().f_back.f_lineno,
                )
                if name not in self._result.keys():
                    self._result[name] = []
                self._result[name].append(toc - tic)
                return result
            else:
                return func(*args, **kwargs)
        return wrapper

    @property
    def enable(self):
        return self._enable

    def set_enable(self):
        self._enable = True
    
    def set_disable(self):
        self._enable = False
    
    @staticmethod
    def _format_cnt(cnt: int) -> str:
        cnt = int(cnt)
        int_str = str(cnt)
        if len(int_str) < 7:
            return (int_str + "x").center(7)
        else:
            return f"{float(cnt):.1e}".center(7)
    
    def clear_all(self):
        for k in self._result.keys():
            self._result[k] = []
    
    def get_report(self, start_path=None, key="func", remove_first=1):
        @dataclass
        class T:
            name: str
            time: float
            func: TimerFuncName

            def __hash__(self) -> int:
                return (self.name, self.time).__hash__()

        function_name_title = "function name"
        max_name_length = len(function_name_title)
        for func_name in self._result.keys():
            max_name_length = max(max_name_length, len(func_name.to_str(start_path)))

        title = f"| {function_name_title.center(max_name_length)} | count | min(ms) | avg(ms) | max(ms) |"
        hline =             f"+-{''.center(max_name_length, '-')}-+-------+---------+---------+---------+"
        str_list = [hline, title, hline]

        str_dict: Dict[T, Tuple[int, float, float, float]] = {}
        for func_name, time_cost in self._result.items():
            if len(time_cost) <= remove_first:
                continue
            
            data = np.array(time_cost)[remove_first:] * 1e3
            func_name_str = func_name.to_str(start_path).ljust(max_name_length)
            data_min = data.min()
            data_mean = data.mean()
            data_max = data.max()
            str_dict[T(func_name_str, data_mean, func_name)] = (len(time_cost), data_min, data_mean, data_max)

        for k in sorted(str_dict.keys(), key=lambda x: getattr(x, key)):
            cnt, m, a, M = str_dict[k]
            str_list.append(f"| {k.name} |{self._format_cnt(cnt)}|{m:.3e}|{a:.3e}|{M:.3e}|")

        str_list.append(hline)
            
        return "\n".join(str_list)


class TiFieldName:
    def __init__(self, prefix: str, filepath: str, lineno: int) -> None:
        self._prefix = prefix
        self._filepath = filepath
        self._lineno = lineno
        self._val = (prefix, filepath, lineno)

    def __hash__(self) -> int:
        return self._val.__hash__()
    
    def __eq__(self, __value: object) -> bool:
        assert isinstance(__value, TiFieldName)
        return self._val == __value._val
    
    def to_str(self, start_path=None) -> str:
        return f"{self._prefix} at file '{os.path.relpath(self._filepath, start_path)}', line {self._lineno}"
    

class TiFieldCreater:
    def __init__(self) -> None:
        self._result: Dict[TiFieldName, List[Tuple[Tuple[int]]]] = {}

    @staticmethod
    def _get_name(prefix):
        return TiFieldName(prefix, inspect.currentframe().f_back.f_back.f_code.co_filename, inspect.currentframe().f_back.f_back.f_lineno)
    
    def _save_result(self, key_str, shape):
        if key_str not in self._result.keys():
            self._result[key_str] = []
        self._result[key_str].append(shape)
    
    def _modify_shape_kwargs(self, kwargs: dict):
        if not isinstance(kwargs["shape"], (list, tuple)):
            kwargs["shape"] = [kwargs["shape"]]
        kwargs["shape"] = list(kwargs["shape"])
        for i in range(len(kwargs["shape"])):
            if not isinstance(kwargs["shape"][i], int):
                kwargs["shape"][i] = int(kwargs["shape"][i])

    def ScalarField(self, dtype, *args, **kwargs) -> ti.ScalarField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("ScalarField"), (tuple(kwargs["shape"]), ))
        return ti.field(dtype=dtype, *args, **kwargs)

    def VectorField(self, n, dtype, *args, **kwargs) -> ti.MatrixField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("VectorField"), (tuple(kwargs["shape"]), (n, )))
        return ti.Vector.field(n=n, dtype=dtype, *args, **kwargs)

    def MatrixField(self, n, m, dtype, *args, **kwargs) -> ti.MatrixField:
        if "shape" in kwargs.keys():
            self._modify_shape_kwargs(kwargs)
            self._save_result(self._get_name("MatrixField"), (tuple(kwargs["shape"]), (n, m)))
        return ti.Matrix.field(n=n, m=m, dtype=dtype, *args, **kwargs)
    
    def StructField(self, cls, **kwargs) -> ti.StructField:
        if "shape" in kwargs.keys():
            if not isinstance(kwargs["shape"], (list, tuple)):
                kwargs["shape"] = (kwargs["shape"], )
            self._save_result(self._get_name("StructField"), (tuple(kwargs["shape"]), cls().get_shape()))
        return cls.field(**kwargs)
    
    def LogSparseField(self, shape):
        if not isinstance(shape, (list, tuple)):
            shape = (shape, )
        self._save_result(self._get_name("SparseField"), (tuple(shape), ))
    
    @property
    def result(self) -> dict:
        return self._result

    @staticmethod
    def _multiple(*args):
        ans = 1
        for x in args:
            ans *= x
        return ans
    
    @staticmethod
    def _calculate_size(field_name: TiFieldName, shape_list: Tuple[Tuple[int]]) -> int:
        if field_name._prefix == "SparseField":
            return 0
        
        size = 0
        for full_shape in shape_list:
            if len(full_shape) == 1:
                shape, = full_shape
                size += TiFieldCreater._multiple(*shape)
            else:
                shape, mn = full_shape
                size += TiFieldCreater._multiple(*shape) * TiFieldCreater._multiple(*mn)
        return size

    def get_report(self, start_path=None, key="size", reverse=True) -> str:
        @dataclass
        class T:
            name: str
            size: float
            cnt: int
            def __hash__(self) -> int:
                return (self.name, self.size, self.cnt).__hash__()
        
        total_size = 0
        str_dict = {}
        for k, v in self._result.items():
            t = T(k.to_str(start_path), TiFieldCreater._calculate_size(k, v), len(v))
            str_dict[t] = v
            total_size += t.size

        str_list = []
        total_field_cnt = 0
        for k in sorted(str_dict.keys(), key=lambda x: getattr(x, key), reverse=reverse):
            total_field_cnt += k.cnt
            str_list.append(f"{k.name}, {k.cnt}x, total size:{k.size} ~ {k.size / total_size * 100:.1f}%, {str_dict[k]}")
        return "\n".join([f"Total fields: {total_field_cnt}x"] + str_list)


GLOBAL_TIMER = Timer()
GLOBAL_CREATER = TiFieldCreater()


@ti.data_oriented
class BaseClass:
    def __init__(self, global_cfg: omegaconf.DictConfig) -> None:
        self._batch_size: int = int(global_cfg.batch_size)
        self._dtype: torch.dtype = getattr(torch, global_cfg.default_float)
        assert isinstance(self._dtype, torch.dtype)
        self._dtype_int: torch.dtype = getattr(torch, global_cfg.default_int)
        assert isinstance(self._dtype_int, torch.dtype)
        self._device: str = str(global_cfg.torch_device)

        self._log_verbose = int(global_cfg.log_verbose)

    @property
    def batch_size(self):
        return self._batch_size
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def dtype_int(self):
        return self._dtype_int
    
    @property
    def device(self):
        return self._device
    
    @property
    def log_verbose(self):
        return self._log_verbose


def unique_name(type_name: str, exist_names: Iterable) -> str:
    assert isinstance(type_name, str)

    exist_idx = [-1]
    type_name_len = len(type_name)
    for name in exist_names:
        if name[:type_name_len] == type_name:
            try:
                idx = int(name[type_name_len:])
                exist_idx.append(idx)
            except Exception:
                pass
    
    return f"{type_name}{max(exist_idx) + 1}"


def get_eps(dtype):
    if dtype == torch.float32:
        return 1e-6
    elif dtype == torch.float64:
        return 1e-14
    else:
        raise NotImplementedError(dtype)
    

def create_zero_7d_pos(batch_size: int, dtype:torch.dtype, device) -> torch.Tensor:
    return torch.tensor([[0., 0., 0., 1., 0., 0., 0.]] * batch_size, dtype=dtype, device=device)


def create_zero_6d_vel(batch_size: int, dtype:torch.dtype, device) -> torch.Tensor:
    return torch.tensor([[0., 0., 0., 0., 0., 0.]] * batch_size, dtype=dtype, device=device)


def create_zero_4x4_eye(batch_size: int, dtype:torch.dtype, device) -> torch.Tensor:
    ret = torch.zeros((batch_size, 4, 4), dtype=dtype, device=device)
    ret[:, [0, 1, 2, 3], [0, 1, 2, 3]] = 1.0
    return ret


def get_raster_points(res=[64, 64, 64], bound=[[-1.0, -1.0, -1.0], [+1.0, +1.0, +1.0]], dtype=np.float32):
    points = np.meshgrid(
        np.linspace(bound[0][0], bound[1][0], res[0]),
        np.linspace(bound[0][1], bound[1][1], res[1]),
        np.linspace(bound[0][2], bound[1][2], res[2])
    )
    points = np.stack(points)
    points = np.swapaxes(points, 1, 2)
    points = points.reshape(3, -1).transpose().astype(dtype)
    return points


@ti.kernel
def _detect_self_intersection_kernel(
    vertices: ti.types.ndarray(dtype=ti.math.vec3), 
    faces: ti.types.ndarray(dtype=ti.math.ivec3), 
    edges: ti.types.ndarray(dtype=ti.math.ivec2),
    intersection_mask: ti.types.ndarray(),
    dx_eps: float) -> bool:
    ret = False
    for fid, eid in ti.ndrange(faces.shape[0], edges.shape[0]):
        intersection_mask[fid, eid] = 0
        v0id, v1id, v2id = faces[fid]
        v3id, v4id = edges[eid]
        edge_on_face = (v3id == v0id or v3id == v1id or v3id == v2id) or \
            (v4id == v0id or v4id == v1id or v4id == v2id)
        if not edge_on_face:
            mat = ti.Matrix.zero(ti.f64, 3, 3)
            mat[:, 0] = ti.cast(vertices[v3id] - vertices[v4id], ti.f64)
            mat[:, 1] = ti.cast(vertices[v1id] - vertices[v0id], ti.f64)
            mat[:, 2] = ti.cast(vertices[v2id] - vertices[v0id], ti.f64)
            xyz_scale = ti.abs(mat).sum() / 9
            mat_det = mat.determinant()
            if ti.abs(mat_det) > (xyz_scale ** 2) * dx_eps:
                right = vertices[v3id] - vertices[v0id]
                left = mat.inverse() @ ti.cast(right, ti.f64)

                a, b, c = 1. - left[1] - left[2], left[1], left[2]
                t = left[0]
                abct = ti.Vector([a, b, c, t], ti.f64)
                zero_f64 = ti.cast(0.0, ti.f64)
                one_f64 = ti.cast(1.0, ti.f64)
                if (zero_f64 < abct).all() and (abct < one_f64).all():
                    ret = True
                    intersection_mask[fid, eid] = 1

    return ret


def detect_self_intersection(mesh: trimesh.Trimesh, dx_eps=1e-7):
    assert mesh.faces.shape[0] * mesh.edges.shape[0] < 1e9
    intersection_mask = np.zeros((mesh.faces.shape[0], mesh.edges.shape[0]), int)
    return bool(_detect_self_intersection_kernel(mesh.vertices, mesh.faces, mesh.edges, intersection_mask, dx_eps)), intersection_mask
