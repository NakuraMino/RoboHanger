import taichi as ti

import json
import os
import sys
import random
import pathlib

import hydra
import omegaconf

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

import lightning.pytorch as pl
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.profilers import AdvancedProfiler, SimpleProfiler, PyTorchProfiler
from lightning.pytorch.callbacks import ModelCheckpoint, ModelSummary, LearningRateMonitor

import robohang.common.utils as utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.funnel.keypoints_unet import (
    make_keypoint_dataset, 
    KeypointsModule,  
)


@hydra.main(config_path="../config/run", config_name=pathlib.Path(__file__).stem, version_base='1.3')
def main(cfg: omegaconf.DictConfig):
    # setup
    utils.init_omegaconf()
    omegaconf.OmegaConf.resolve(cfg)
    cfg = utils.resolve_overwrite(cfg)
    omegaconf.OmegaConf.save(cfg, os.path.join(os.getcwd(), ".hydra", "resolved.yaml"))

    # init numpy, pytorch, taichi
    torch.random.manual_seed(cfg.misc.seed)
    np.random.seed(cfg.misc.seed)
    random.seed(cfg.misc.seed)
    torch.set_float32_matmul_precision(cfg.misc.hardware.precision)
    ti.init(
        arch=getattr(ti, cfg.misc.taichi.device), 
        default_fp=ti.f32, default_ip=ti.i32, device_memory_GB=cfg.misc.taichi.device_memory_GB, 
        offline_cache=True, fast_math=False, debug=False
    )

    # init logger
    with open(os.path.join("command_line.json"), "w") as f_obj:
        json.dump(obj=dict(
            command_line=" ".join(sys.argv),
            run_command_dir=utils.get_path_handler()("."),
        ), fp=f_obj, indent=4)

    logger = pl_loggers.TensorBoardLogger(os.getcwd(), version="")
    if utils.ddp_is_rank_0():
        print("logger.log_dir", logger.log_dir)
        os.makedirs(logger.log_dir)
    profiler_name2class = {"Advanced": AdvancedProfiler, "Simple": SimpleProfiler, "PyTorch": PyTorchProfiler}
    profiler = profiler_name2class[cfg.misc.debug.profiler](dirpath=logger.log_dir, filename="perf_logs")

    # dataset and dataloader
    trds, vlds = make_keypoint_dataset(
        [utils.get_path_handler()(s) for s in cfg.train.path.data_paths], 
        cfg.data.common.valid_size, 
        cfg.data.make, 
        cfg.data.dataset,
    )
    ti.reset(), ti.init(ti.cpu)
    trdl = DataLoader(
        trds, 
        batch_size=cfg.data.common.batch_size, 
        num_workers=cfg.data.common.num_workers, 
        drop_last=cfg.data.common.drop_last, 
        shuffle=True,
    )
    vldl = DataLoader(
        vlds, 
        batch_size=cfg.data.common.batch_size, 
        num_workers=cfg.data.common.num_workers, 
        drop_last=cfg.data.common.drop_last, 
        shuffle=False,
    )

    # training
    ckpt_path = cfg.train.path.ckpt

    # init model
    if ckpt_path is not None:
        model = KeypointsModule.load_from_checkpoint(
            utils.get_path_handler()(ckpt_path), 
            model_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.model), 
            learn_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.learn),
        )
    else:
        model = KeypointsModule(
            model_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.model), 
            learn_kwargs=omegaconf.OmegaConf.to_container(cfg.pl.learn),
        )

    # init trainer
    p = learn_utils.get_profiler()
    trainer_kwargs = {
        "accelerator": "cuda" if cfg.misc.hardware.cuda else "cpu",
        "devices": cfg.misc.hardware.gpuids if cfg.misc.hardware.gpuids else "auto",
        "strategy": "ddp",
        "max_steps": cfg.train.cfg.max_steps + cfg.misc.step_offset,
        "logger": logger,
        "profiler": profiler,
        "limit_train_batches": cfg.train.cfg.limit_train_batches,
        "limit_val_batches": cfg.train.cfg.limit_val_batches,
        "log_every_n_steps": cfg.train.cfg.log_every_n_steps,
        "val_check_interval": cfg.train.cfg.val_check_interval,
        "check_val_every_n_epoch": None,
        "callbacks": [
            ModelCheckpoint(
                every_n_train_steps=cfg.train.cfg.ckpt_every_n_steps, 
                save_top_k=-1, 
                dirpath=f'ckpt', 
                filename='epoch_{epoch}_step_{step}', 
                auto_insert_metric_name=False
            ), 
            ModelSummary(max_depth=cfg.misc.debug.model_summary_max_depth), 
            LearningRateMonitor(logging_interval='step'),
            learn_utils.TorchProfilerCallback(p),
        ],
    }
    trainer = pl.Trainer(**trainer_kwargs)

    with p:
        # train
        if cfg.run.mode == "train":
            print("start fitting model...")
            ckpt_path = utils.get_path_handler()(ckpt_path) if ckpt_path is not None else None
            trainer.fit(
                model=model,
                train_dataloaders=trdl,
                val_dataloaders=vldl,
                ckpt_path=ckpt_path,
            )
        elif cfg.run.mode == "eval":
            print("start evaluating model...")
            trainer.validate(
                model=model,
                dataloaders=vldl,
            )
        else:
            raise ValueError(cfg.run.mode)


if __name__ == "__main__":
    main()