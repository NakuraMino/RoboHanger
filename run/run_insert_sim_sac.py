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
from robohang.agent.base_agent import GalbotOneAgent
from robohang.policy.insert.insert_gym import InsertGym
from robohang.policy.insert.insert_learn_sac import InsertPolicySAC
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
    insert_gym = InsertGym(sim_env, agent, cfg.insert_gym)
    insert_policy = InsertPolicySAC(insert_gym, cfg.insert_policy)

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

    def export_all(action_and_info: Optional[dict]):
        def export_json(path, info):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path, "w") as f_obj:
                json.dump(info, fp=f_obj, indent=4)
        def export_npy(path, info):
            os.makedirs(os.path.dirname(path), exist_ok=True)
            np.save(path, info)
        # score
        export_json(os.path.join(exporter.get_trajectory_path(), "score", exporter.get_step_str() + ".json"), insert_gym.get_score())
        # sim_error
        export_json(os.path.join(exporter.get_trajectory_path(), "sim_error", exporter.get_step_str() + ".json"), sim_env.get_sim_error())
        # state
        export_npy(os.path.join(exporter.get_trajectory_path(), "state", exporter.get_step_str() + ".npy"), insert_gym.get_state())
        # action
        if action_and_info is not None:
            export_npy(os.path.join(exporter.get_trajectory_path(), "action", exporter.get_step_str() + ".npy"), utils.torch_dict_to_numpy_dict(action_and_info["action"]))
            export_json(os.path.join(exporter.get_trajectory_path(), "action", exporter.get_step_str() + ".json"), utils.torch_dict_to_list_dict(action_and_info["info"]))
        
        os.makedirs("timer", exist_ok=True)
        with open(f"timer/{exporter.get_trajectory_folder_name()}_{exporter.get_step_str()}.txt", "w") as f_obj:
            f_obj.write(sim_utils.GLOBAL_TIMER.get_report())
            sim_utils.GLOBAL_TIMER.clear_all()

        exporter.export_policy_obs()

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
        
        action_and_info = insert_policy.get_press_action(info=dict(log_path=os.path.join(exporter.get_trajectory_path(), "pred", "0")))
        export_all(action_and_info)
        insert_gym.primitive_press(**(action_and_info["action"]), callbacks=callbacks)

        action_and_info = insert_policy.get_lift_action(info=dict(log_path=os.path.join(exporter.get_trajectory_path(), "pred", "1")))
        export_all(action_and_info)
        insert_gym.primitive_lift(**(action_and_info["action"]), callbacks=callbacks)

        action_and_info = insert_policy.get_drag_action(info=dict(log_path=os.path.join(exporter.get_trajectory_path(), "pred", "2")))
        export_all(action_and_info)
        insert_gym.primitive_drag(**(action_and_info["action"]), callbacks=callbacks)

        action_and_info = insert_policy.get_rotate_action(info=dict(log_path=os.path.join(exporter.get_trajectory_path(), "pred", "3")))
        export_all(action_and_info)
        insert_gym.primitive_rotate(**(action_and_info["action"]), callbacks=callbacks)

        export_all(None)
        insert_gym.reset()
        exporter.update_traj_idx()


if __name__ == "__main__":
    main()