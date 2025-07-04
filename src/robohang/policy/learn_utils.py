import os
from typing import List, Optional, Union, Dict, Tuple, Any, Type, Callable
import re
from dataclasses import dataclass, asdict
import threading
import multiprocessing
import concurrent.futures
import time

import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import omegaconf

import robohang.common.utils as utils

from torch.utils.tensorboard import SummaryWriter

from lightning.pytorch.callbacks import Callback
from lightning import LightningModule, Trainer
from lightning.pytorch.utilities.types import STEP_OUTPUT

def parse_size(valid_size_raw: float, total_size: int) -> int:
    valid_size_raw = float(valid_size_raw)
    total_size = int(total_size)

    if 0 <= valid_size_raw and valid_size_raw <= 1:
        valid_size = int(total_size * valid_size_raw)
    else:
        valid_size = int(valid_size_raw)
    assert valid_size > 0 and valid_size < total_size, f"{valid_size_raw} {total_size} {valid_size}"

    return valid_size


@dataclass   
class NodeInfo:
    file_path: str
    parent_node: 'DataTree'


class DataTree:
    def __init__(
        self, 
        base_dir: str, 
        basename_patterns: List[str], 
        parent_node: Optional['DataTree'],
        is_exclude_path: Optional[Callable[[str], bool]],
    ) -> None:
        assert isinstance(base_dir, str)
        assert isinstance(basename_patterns, list)
        if parent_node is not None:
            assert isinstance(parent_node, DataTree)

        self.base_dir = str(os.path.abspath(base_dir))
        self.basename_patterns = list(basename_patterns)
        self.parent_node = parent_node
        self.is_exclude_path = is_exclude_path
        self.children: List[DataTree] = []
        self.node_info: Dict[str, List[NodeInfo]] = {pattern: [] for pattern in self.basename_patterns}
        self.accumulate: Dict[str, np.ndarray] = {}
        
        self._build()

    def _build(self):
        accumulate = {pattern: [0] for pattern in self.basename_patterns}
        for child_path_rel in sorted(os.listdir(self.base_dir)):
            child_path = os.path.join(self.base_dir, child_path_rel)
            if callable(self.is_exclude_path):
                if self.is_exclude_path(child_path):
                    continue
            if os.path.isfile(child_path):
                basename = os.path.basename(child_path)
                for pattern in self.basename_patterns:
                    if re.search(pattern, basename) is not None:
                        self.node_info[pattern].append(NodeInfo(child_path, self))
            else:
                self.children.append(DataTree(child_path, self.basename_patterns, self, self.is_exclude_path))
                for pattern in self.basename_patterns:
                    accumulate[pattern].append(accumulate[pattern][-1] + self.children[-1]._get_tree_size(pattern))
        
        for pattern in self.basename_patterns:
            self.accumulate[pattern] = np.array(accumulate[pattern]) + self._get_node_size(pattern)

    def _get_node_size(self, pattern: str) -> int:
        return len(self.node_info[pattern])
    
    def _get_tree_size(self, pattern: str) -> int:
        return int(self.accumulate[pattern][-1])
    
    def _get_item(self, pattern: str, idx: int) -> NodeInfo:
        assert 0 <= idx < self._get_tree_size(pattern), f"not 0 <= {idx} < {self._get_tree_size(pattern)}"
        which_child = np.searchsorted(self.accumulate[pattern], idx, side="right") - 1
        if which_child == -1:
            return self.node_info[pattern][idx]
        else:
            return self.children[which_child]._get_item(pattern, idx - self.accumulate[pattern][which_child])
        
    def __getitem__(self, __index: Union[Tuple[str, int], str]) -> Union[NodeInfo, int]:
        if isinstance(__index, tuple):
            pattern, idx = __index
            return self._get_item(pattern, idx)
        elif isinstance(__index, str):
            pattern = __index
            return self._get_tree_size(pattern)
        else:
            raise IndexError(__index)
    
    def get_node_size(self, pattern: str) -> int:
        return self._get_node_size(pattern)
    
    def get_tree_size(self, pattern: str) -> int:
        return self._get_tree_size(pattern)
    
    def __repr__(self) -> str:
        return f"DataTree(base_dir='{self.base_dir}')"


class DataForest:
    def __init__(
        self, 
        base_dirs: List[str],
        basename_patterns: List[str],
        is_exclude_path: Optional[Callable[[str], bool]]=None,
    ) -> None:
        assert isinstance(base_dirs, list)
        assert isinstance(basename_patterns, list)

        self.base_dirs = list(base_dirs)
        self.trees = [DataTree(base_dir, basename_patterns, None, is_exclude_path) for base_dir in base_dirs]
        self.basename_patterns = list(basename_patterns)
        self.accumulate: Dict[str, np.ndarray] = {}

        self._build()

    def _build(self):
        accumulate = {pattern: [0] for pattern in self.basename_patterns}
        for tree in self.trees:
            for pattern in self.basename_patterns:
                accumulate[pattern].append(accumulate[pattern][-1] + tree[pattern])

        for pattern in self.basename_patterns:
            self.accumulate[pattern] = np.array(accumulate[pattern])

    def _get_forest_size(self, pattern: str):
        return int(self.accumulate[pattern][-1])

    def _get_item(self, pattern: str, idx: int) -> NodeInfo:
        assert 0 <= idx < self._get_forest_size(pattern), f"not 0 <= {idx} < {self._get_forest_size(pattern)}"
        which_child = np.searchsorted(self.accumulate[pattern], idx, side="right") - 1
        return self.trees[which_child]._get_item(pattern, idx - self.accumulate[pattern][which_child])
    
    def __getitem__(self, __index: Union[Tuple[str, int], str]) -> Union[NodeInfo, int]:
        if isinstance(__index, tuple):
            pattern, idx = __index
            return self._get_item(pattern, idx)
        elif isinstance(__index, str):
            pattern = __index
            return self._get_forest_size(pattern)
        else:
            raise IndexError(__index)
        
    def get_forest_size(self, pattern: str) -> int:
        return self._get_forest_size(pattern)
    
    def get_item(self, pattern: str, idx: int) -> NodeInfo:
        return self._get_item(pattern, idx)
    
    def __repr__(self) -> str:
        return f"DataForest(base_dirs='{self.base_dirs}')"


def get_profiler(wait=1, warmup=1, active=2, repeat=1):
    return torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        schedule=torch.profiler.schedule(wait=wait, warmup=warmup, active=active, repeat=repeat),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile'),
        record_shapes=True,
        with_stack=True,
    )


class TorchProfilerCallback(Callback):
    def __init__(self, profiler: torch.profiler.profile) -> None:
        super().__init__()
        self.profiler = profiler

    def on_train_batch_end(self, trainer: Trainer, pl_module: LightningModule, outputs: STEP_OUTPUT, batch: Any, batch_idx: int) -> None:
        self.profiler.step()


def conv(img: np.ndarray, kernel: np.ndarray):
    H, W = img.shape
    HC, WC = kernel.shape
    L = WC // 2
    R = WC - L - 1
    T = HC // 2
    B = HC - T - 1

    ans = np.zeros((H, W, HC, WC))
    i = np.arange(H)[:, None, None, None]
    j = np.arange(W)[None, :, None, None]
    k = np.arange(HC)[None, None, :, None]
    l = np.arange(WC)[None, None, None, :]
    ans[i, j, k, l] = np.pad(img, ((T, B), (L, R)), mode="edge")[i + k, j + l] * kernel[k, l]
    return ans.sum(axis=(2, 3))


def data_augmentation(depth_raw: np.ndarray, mask_raw: np.ndarray, cfg: omegaconf.DictConfig):
    info = dict()
    depth = depth_raw.copy()
    mask = mask_raw.copy()
    H, W = depth.shape

    if len(mask.shape) == 2:
        assert mask.shape == (H, W), mask.shape
    elif len(mask.shape) == 3:
        assert mask.shape[1] == H and mask.shape[2] == W, mask.shape
    else:
        raise ValueError(mask.shape)
    
    depth_mean = np.mean(depth)
    depth_scale = utils.map_01_ab(np.random.random(), cfg.depth_scale[0], cfg.depth_scale[1])
    depth = depth_mean + (depth - depth_mean) * depth_scale
    
    kernel = np.array([
        [1., 2., 1.],
        [2., 4., 2.],
        [1., 2., 1.],
    ])
    kernel /= np.sum(kernel)
    depth += (
        np.linspace(-1., 1., H)[:, None] * (np.random.rand() * 2. - 1.) * cfg.tilt +
        np.linspace(-1., 1., W)[None, :] * (np.random.rand() * 2. - 1.) * cfg.tilt
    ) + (
        conv(np.random.randn(H, W), kernel) if cfg.use_conv_on_noise
        else np.random.randn(H, W)
    ) * cfg.noise_std

    if cfg.random_flip_mask:
        info["mask_unflip"] = mask.copy()
        flip_mask_idx = np.where(np.random.binomial(1, cfg.flip_mask_prob, size=mask.shape) == 1)
        mask[flip_mask_idx] = 1. - mask[flip_mask_idx]
    return depth, mask, info


def one_hot(data: torch.Tensor, size: torch.Size, dtype: torch.dtype, device: torch.device):
    """
    Args:
        data: [B, 2]
        size: == B, H, W
    Return:
        one_hot: [B, H, W]
    """
    ret = torch.zeros(size=size, dtype=dtype, device=device)
    B = data.shape[0]
    ret[torch.arange(B, device=device), data[:, 0], data[:, 1]] = 1
    return ret


def out_of_action_space_to_min(dense: torch.Tensor, action_space: torch.Tensor):
    """
    Args:
    - dense: [B, H, W]
    - action_space: [B, H, W]

    Return:
    - dense_masked: clone(), [B, H, W]
    """
    B, H, W = dense.shape
    assert action_space.shape == dense.shape, f"{dense.shape} {action_space.shape}"
    dense = torch.where(action_space == 0., torch.inf, dense)
    dense = torch.where(action_space == 0., dense.view(B, -1).min(dim=-1).values.view(B, 1, 1), dense)
    return dense


def plot_wrap_fig(
    denses: List[np.ndarray], 
    titles: List[Union[str, List[str]]], 
    colorbars: List[Optional[str]], 
    plot_batch_size: int,
    width_unit=4.,
    height_unit=6.,
):
    plot_fig_num = len(denses)
    fig = plt.figure(figsize=(width_unit * plot_fig_num, height_unit * plot_batch_size))
    spec = fig.add_gridspec(
        nrows=plot_batch_size, 
        ncols=3 * plot_fig_num, 
        width_ratios=[6, 1, 2] * plot_fig_num, 
        height_ratios=[1] * plot_batch_size,
    )

    for batch_idx in range(plot_batch_size):
        for img_idx, img_b in enumerate(denses):
            img = img_b[batch_idx]
            ax = fig.add_subplot(spec[batch_idx, 3 * img_idx])
            if colorbars[img_idx] is not None:
                cmap = plt.get_cmap(colorbars[img_idx])
                ax.imshow(img, cmap=cmap)
            else:
                ax.imshow(img)
            if isinstance(titles[img_idx], str):
                ax.set_title(titles[img_idx])
            else:
                assert isinstance(titles[img_idx], list)
                assert isinstance(titles[img_idx][batch_idx], str)
                ax.set_title(titles[img_idx][batch_idx])
            if colorbars[img_idx] is not None:
                vmin, vmax = img.min(), img.max()
                cax = fig.add_subplot(spec[batch_idx, 3 * img_idx + 1])
                cbar = fig.colorbar(
                    mappable=ScalarMappable(norm=plt.Normalize(vmin, vmax), cmap=cmap), 
                    cax=cax,
                )

    return fig


def plot_wrap(denses: List[np.ndarray], tag: str, titles: List[Union[str, List[str]]], colorbars: List[Optional[str]],
              plot_batch_size: int, global_step: int, writer: SummaryWriter):
    """
    Args:
        - denses: List of [B, H, W]
        - tag: str
        - titles: List of str
    """
    assert len(denses) == len(titles) == len(colorbars), f"{len(denses)} {len(titles)} {len(colorbars)}"
    fig = plot_wrap_fig(denses, titles, colorbars, plot_batch_size)
    writer.add_figure(tag, fig, global_step=global_step)
    writer.close()
    plt.close()


def annotate_action(denseRGB: np.ndarray, actions: List[np.ndarray], colors: List[np.ndarray], widths: List[int]):
    assert len(actions) == len(colors) == len(widths), f"{len(actions)} {len(colors)} {len(widths)}"
    denseRGB = denseRGB.copy()
    for batch_idx, d in enumerate(denseRGB):
        vmin = d.min()
        vmax = d.max()
        d -= vmin
        d /= vmax - vmin
        for action, color, width in zip(actions, colors, widths):
            width_half = width // 2 
            assert len(action.shape) == 2, action.shape
            for i in range(action[batch_idx, 0] - width_half, action[batch_idx, 0] - width_half + width):
                for j in range(action[batch_idx, 1] - width_half, action[batch_idx, 1] - width_half + width):
                    if (0 <= i < d.shape[0]) and (0 <= j < d.shape[1]):
                        d[i, j] = color
    return denseRGB


def npstr2str(s):
    return str(s, "utf-8")


class StrArray:
    """https://github.com/pytorch/pytorch/issues/13246#issuecomment-617140519"""
    def __init__(self, x: List[str]):
        self._v, self._o = self.pack_sequences([self.string_to_sequence(s) for s in x])
    
    def __getitem__(self, i):
        seq = self.unpack_sequence(self._v, self._o, i)
        string = self.sequence_to_string(seq)
        return string
    
    @staticmethod
    def string_to_sequence(s: str, dtype=np.int32) -> np.ndarray:
        return np.array([ord(c) for c in s], dtype=dtype)

    @staticmethod
    def sequence_to_string(seq: np.ndarray) -> str:
        return ''.join([chr(c) for c in seq])

    @staticmethod
    def pack_sequences(seqs: Union[np.ndarray, list]) -> Tuple[np.ndarray, np.ndarray]:
        values = np.concatenate(seqs, axis=0)
        offsets = np.cumsum([len(s) for s in seqs])
        return values, offsets

    @staticmethod
    def unpack_sequence(values: np.ndarray, offsets: np.ndarray, index: int) -> np.ndarray:
        off1 = offsets[index]
        if index > 0:
            off0 = offsets[index - 1]
        elif index == 0:
            off0 = 0
        else:
            raise ValueError(index)
        return values[off0:off1]


class DataclassArray:
    def __init__(self, l: List[Any], default_int=np.int64, default_float=np.float64):
        self._data: Dict[str, List[Any]] = {}
        self._type: Dict[str, Type] = {}
        self._dataclass = None
        for x in l:
            if self._dataclass is None:
                self._dataclass = type(x)
            else:
                assert self._dataclass == type(x)
            for k, v in type(x).__annotations__.items():
                if k not in self._data.keys():
                    self._data[k] = []
                    self._type[k] = v
                self._data[k].append(v(getattr(x, k)))
        self._default_int = default_int
        self._default_float = default_float
        
        for k in self._data.keys():
            if self._type[k] == str:
                self._data[k] = np.array(self._data[k]).astype(np.string_)
            elif self._type[k] == int:
                self._data[k] = np.array(self._data[k]).astype(np.int64)
            elif self._type[k] == float:
                self._data[k] = np.array(self._data[k]).astype(np.float64)
            else:
                self._data[k] = np.array(self._data[k])
    
    '''def __getattr__(self, name: str) -> np.ndarray:
        return self._data[name]
    
    @staticmethod
    def to_str(bstr):
        return 
    
    def keys(self):
        return self._data.keys()'''
    
    def index(self, idx: int):
        d = {}
        for k in self._data.keys():
            if self._type[k] == str:
                d[k] = str(self._data[k][idx], "utf-8")
            else:
                d[k] = self._data[k][idx]
        return self._dataclass(**d)


'''class DaemonProcessPoolExecutor(concurrent.futures.ProcessPoolExecutor):
    def _adjust_thread_count(self):
        if len(self._threads) < self._max_workers:
            thread = threading.Thread()
            thread.daemon = True
            thread._target = self._worker
            thread._name = f"ThreadPoolExecutor-{self._thread_name_prefix or ''}{len(self._threads)}"
            thread.start()
            self._threads.add(thread)
'''

class MultiProcessLauncher:
    def __init__(self, num_worker: int, sleep_time=1e-3) -> None:
        self.num_worker = int(num_worker)
        self.sleep_time = float(sleep_time)
        self.processes: List[Optional[multiprocessing.Process]] = [None for _ in range(self.num_worker)]
    
    def find_available_process(self):
        for pidx in range(self.num_worker):
            p = self.processes[pidx]
            if p is None:
                return pidx
            elif not p.is_alive():
                p.join()
                self.processes[pidx] = None
                return pidx
        return None
    
    def launch_worker(self, worker: Callable, args=(), kwargs=None,):
        while True:
            pidx = self.find_available_process()
            if pidx is None:
                time.sleep(self.sleep_time)
            else:
                break
        p = multiprocessing.Process(
            target=worker,
            args=args,
            kwargs=kwargs if kwargs is not None else {},
            daemon=True
        )
        p.start()
        self.processes[pidx] = p
    
    def join_all(self):
        for p in self.processes:
            p.join()
