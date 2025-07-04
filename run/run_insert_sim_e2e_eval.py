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
import robohang.sim.sim_utils as sim_utils
from robohang.env.sim_env import SimEnv
from robohang.env.sapien_renderer import CameraProperty
from robohang.agent.base_agent import GalbotOneAgent
from robohang.policy.insert.insert_gym_e2e import InsertGymE2E, ExporterE2E
from robohang.policy.insert.insert_policy_e2e import InsertPolicyE2E, InsertPolicyDebug
from robohang.policy.insert.insert_learn_act import InsertPolicyACT
from robohang.policy.insert.insert_learn_dfp import InsertPolicyDfP
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

    # init env
    sim_env = SimEnv(cfg.sim_env, cfg.glb_cfg)
    agent = GalbotOneAgent(sim_env, cfg.agent)
    insert_gym = InsertGymE2E(sim_env, agent, cfg.insert_gym)
    insert_policy_cls = dict(act=InsertPolicyACT, dfp=InsertPolicyDfP, debug=InsertPolicyDebug)[cfg.insert_policy.name]
    insert_policy: InsertPolicyE2E = insert_policy_cls(insert_gym, getattr(cfg.insert_policy, cfg.insert_policy.name))

    # init exporter
    output_cfg = cfg.output
    exporter = ObservationExporter(
        sim_env=sim_env, 
        agent=agent, 
        export_cfg=output_cfg.export, 
        total_traj=output_cfg.total_traj, 
        total_step=5, 
    )
    if not (output_cfg.collect_mode):
        callbacks.append(exporter.callback_side_view)
    exp_e2e = ExporterE2E(insert_gym)

    def export_misc(traj_path: str):
        os.makedirs(traj_path, exist_ok=True)
        with open(os.path.join(traj_path, "misc.json"), "w") as f_obj:
            json.dump(
                dict(
                    batch_size=sim_env.batch_size,
                    total_traj=output_cfg.total_traj,
                    garment_keypoints=sim_env.garment_keypoints,
                ), f_obj, indent=4,
            )

    # main loop
    for traj_idx in tqdm.tqdm(range(output_cfg.total_traj), dynamic_ncols=True):
        export_misc(exporter.get_trajectory_path())
        insert_gym.domain_randomize()
        if not cfg.control.skip_init_garment:
            insert_gym.primitive_init_garment(callbacks)
        elif cfg.control.state_path is not None:
            insert_gym.set_state(np.load(utils.get_path_handler()(cfg.control.state_path), allow_pickle=True).item())
        # export state to debug
        full_state_path = os.path.join(exporter.get_trajectory_path(), "init_state.npy")
        np.save(full_state_path, insert_gym.get_state(), allow_pickle=True)
        
        # export at first
        exp_e2e.export()
        # rollout
        for _ in range(insert_policy.max_step):
            insert_gym.e2e_step(
                action=insert_policy.get_action(), 
                steps_in_sim=insert_policy.steps_in_sim,
                callbacks=callbacks + ([exp_e2e.callback] if output_cfg.e2e_export_obs else [])
            )
        insert_policy.reset()
        exp_e2e.update_traj_idx()

        insert_gym.reset()
        exporter.update_traj_idx()


if __name__ == "__main__":
    main()