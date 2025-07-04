import logging
logger = logging.getLogger(__name__)

import copy
from typing import Dict, Literal, Any, List, Callable, Tuple
import time
import math
import os

import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2

import omegaconf

import robohang.common.utils as utils
import robohang.sim.sim_utils as sim_utils
import robohang.sim.so3 as so3
import robohang.policy.policy_utils as policy_utils
import robohang.policy.learn_utils as learn_utils
from robohang.policy.policy_utils import MetaInfo
from robohang.policy.insert.insert_gym_e2e import InsertGymE2E, ROBOT_DIM


class InsertPolicyE2E:
    def __init__(self, insert_gym: InsertGymE2E, policy_cfg: omegaconf.DictConfig) -> None:
        self._policy_cfg = copy.deepcopy(policy_cfg)
        self._insert_gym = insert_gym

        self.steps_in_sim = int(policy_cfg.steps_in_sim)
        self.max_step = int(policy_cfg.max_step)

        self.dtype = self._insert_gym.dtype
        self.dtype_int = self._insert_gym.dtype_int
        self.device = self._insert_gym.device
        self.batch_size = self._insert_gym.batch_size

    def get_action(self) -> torch.Tensor:
        """return random action"""

        action = torch.zeros((self.batch_size, ROBOT_DIM), dtype=self.dtype, device=self.device)
        action[:, :2] = torch.randint(0, 3, (self.batch_size, 2), dtype=self.dtype, device=self.device)
        action[:, 2:] = torch.randn((self.batch_size, ROBOT_DIM - 2), dtype=self.dtype, device=self.device) * 0.1

        return action

    def reset(self):
        return


class InsertPolicyDebug(InsertPolicyE2E):
    def __init__(self, insert_gym: InsertGymE2E, policy_cfg: omegaconf.DictConfig) -> None:
        super().__init__(insert_gym, policy_cfg)
        
        self.current_step = 0
        self.debug_action_dir = utils.get_path_handler()(str(policy_cfg.debug_action_dir))

    def get_action(self) -> torch.Tensor:
        action = torch.zeros((self.batch_size, ROBOT_DIM), dtype=self.dtype, device=self.device)
        for batch_idx in range(self.batch_size):
            action_idx = self.current_step - batch_idx
            if action_idx >= 0:
                action[batch_idx, :] = torch.tensor(np.load(
                    os.path.join(self.debug_action_dir, str(batch_idx), "action", f"{action_idx}.npy")
                ))
        self.current_step = min(self.current_step + 1, 139)

        return action