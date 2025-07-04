import logging
logger = logging.getLogger(__name__)

import taichi as ti

import os
import json
from typing import List, Callable, Optional, Literal, Dict

import torch
import numpy as np
import trimesh
import trimesh.transformations as tra
from PIL import Image

import sapien.core as sapien
import omegaconf

import batch_urdf
import robohang.common.utils as utils
from robohang.sim.sim import Sim
from robohang.sim.env import Env
import robohang.sim.so3 as so3
from robohang.agent.base_agent import BaseAgent
from robohang.env.sim_env import SimEnv
from robohang.env.sapien_renderer import CameraProperty
from robohang.policy.insert.insert_gym import InsertGym

ROBOT_DIM = 2 + 7 + 7 + 4

GRASP_NONE = 0
GRASP_CLOTHES = 1
GRASP_HANGER = 2


def camera_intrinsic_to_property(intrinsic: np.ndarray, h: int, w: int):
    return CameraProperty(width=w, height=h, fx=intrinsic[0, 0], fy=intrinsic[1, 1], cx=intrinsic[0, 2], cy=intrinsic[1, 2], skew=intrinsic[0, 1])


def fix_joint_cfg(
    urdf: batch_urdf.URDF, 
    target_cfg: Dict[str, torch.Tensor], 
    table_z: torch.Tensor, 
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int, 
    hanger_transform: Dict[str, torch.Tensor]=None, 
    ee_str_list: Optional[List[str]]=None,
    ik_solver_cfg: Optional[omegaconf.DictConfig]=None,
) -> Dict[str, torch.Tensor]:
    if ee_str_list is None:
        ee_str_list = [
            ("left_gripper_tcp_link", None), 
            ("right_gripper_tcp_link", None),
        ]
    if ik_solver_cfg is None:
        ik_solver_cfg = omegaconf.DictConfig(dict(
            max_iter=64,
            fix_joint=["leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4"]
        ))
    
    mask = torch.tensor([0., 0., 0., 0.] * 2 + [0., 0., 0., 1.] + [0., 0., 0., 1.], dtype=dtype, device=device) # (e'_x)_z = 0
    mat = torch.eye(4, dtype=dtype, device=device).repeat(batch_size, 1, 1)
    mat[:, 2, 3] = table_z
    translation_z_idx = 2 * 4 + 3
    def get_mat(link_transform_map, ee_str, endpoint) -> torch.Tensor:
        mat = link_transform_map[ee_str] # [?, 4, 4]
        if endpoint is None:
            return mat
        else:
            return mat @ hanger_transform[endpoint][None, :, :] # [?, 4, 4]
    def mask_and_clamp(err_mat: torch.Tensor):
        err_mat[:, translation_z_idx] *= (err_mat[:, translation_z_idx] < 0.).float()
        return err_mat * mask
    def err_func(link_transform_map: Dict[str, torch.Tensor]):
        ans = None
        for ee_str, endpoint in ee_str_list:
            curr_mat4 = get_mat(link_transform_map, ee_str, endpoint) # [B*16, 4, 4]
            curr_mat4 = curr_mat4.view(batch_size, 16, 16)
            err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(batch_size, 16) # [B, 16]
            err_mat = mask_and_clamp(err_mat)
            ans = (err_mat) if (ans is None) else (ans + err_mat)
        return ans
    def loss_func(link_transform_map: Dict[str, torch.Tensor]):
        ans = None
        for ee_str, endpoint in ee_str_list:
            curr_mat4 = get_mat(link_transform_map, ee_str, endpoint)
            err_mat = (curr_mat4 - mat).view(batch_size, 16) # [B, 16]
            err_mat = mask_and_clamp(err_mat)
            loss = torch.sum(torch.square(err_mat), dim=1) # [B, ]
            ans = (loss) if (ans is None) else (ans + loss)
        return ans
    def to_full_cfg(cfg: Dict[str, torch.Tensor]):
        if cfg is None:
            return None
        full_cfg = {k: v.clone() for k, v in urdf.cfg.items()}
        for k, v in cfg.items():
            full_cfg[k][...] = v
        return full_cfg
    new_cfg, info = urdf.inverse_kinematics_optimize(
        err_func=err_func,
        loss_func=loss_func,
        init_cfg=to_full_cfg(target_cfg),
        **(ik_solver_cfg),
    )
    logger.info(f"fix_joint_cfg:{info}")
    return new_cfg


def move_left_to_hanger(
    urdf: batch_urdf.URDF, 
    target_cfg: Dict[str, torch.Tensor], 
    dtype: torch.dtype,
    device: torch.device,
    batch_size: int, 
    hanger_mat: torch.Tensor,
    hanger_transform: Dict[str, torch.Tensor], 
    gripper_action: torch.Tensor,
    gripper_state: torch.Tensor,
    ik_solver_cfg: Optional[omegaconf.DictConfig]=None,
):
    if ik_solver_cfg is None:
        ik_solver_cfg = omegaconf.DictConfig(dict(
            max_iter=64,
            fix_joint=["leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4"]
        ))

    assert gripper_action.shape == (batch_size, 2), gripper_action.shape
    assert gripper_state.shape == (batch_size, 2), gripper_state.shape
    left_hand_idx = 0
    mask = torch.logical_and(gripper_action[:, left_hand_idx] == 2., gripper_state[:, left_hand_idx] == 0.).to(dtype=dtype, device=device)

    target_link = "left_gripper_tcp_link"
    def err_func(link_transform_map: Dict[str, torch.Tensor]):
        curr_mat4 = link_transform_map[target_link].view(batch_size, -1, 4, 4) @ hanger_transform["left"].view(1, 1, 4, 4) # [B, 16, 4, 4]
        return (
            curr_mat4.view(batch_size, 16, 16)[:, torch.arange(16), torch.arange(16)] - 
            hanger_mat.view(batch_size, 16)
        ) * mask[:, None] # [B, 16]
    def loss_func(link_transform_map: Dict[str, torch.Tensor]):
        curr_mat4 = link_transform_map[target_link].view(batch_size, 4, 4) @ hanger_transform["left"].view(1, 4, 4) # [B, 4, 4]
        return torch.sum(torch.square(curr_mat4 - hanger_mat) * mask[:, None, None], dim=(1, 2)) # [B, ]

    def to_full_cfg(cfg: Dict[str, torch.Tensor]):
        if cfg is None:
            return None
        full_cfg = {k: v.clone() for k, v in urdf.cfg.items()}
        for k, v in cfg.items():
            full_cfg[k][...] = v
        return full_cfg
    new_cfg, info = urdf.inverse_kinematics_optimize(
        err_func=err_func,
        loss_func=loss_func,
        init_cfg=to_full_cfg(target_cfg),
        **(ik_solver_cfg),
    )
    logger.info(f"move_left_to_hanger:{info}")
    return new_cfg


class InsertGymE2E(InsertGym):
    def __init__(self, sim_env: SimEnv, agent: BaseAgent, gym_cfg: omegaconf.DictConfig) -> None:
        super().__init__(sim_env, agent, gym_cfg)

        self._hanger_transform = dict(
            right = torch.tensor(tra.euler_matrix(*(self._press_cfg.hanger_rpy.transform)), dtype=self.dtype, device=self.device),
            left = torch.tensor(tra.euler_matrix(*(self._drag_cfg.hanger_rpy.transform)), dtype=self.dtype, device=self.device),
        )
        self._min_h = float(self._parameter.min_h)

    def _take_picture(self, camera_prop: CameraProperty, camera_pose: List[float], batch_idx: int, show_gripper=True):
        def invert_face(m: trimesh.Trimesh):
            m.invert()
            return m
        # save obj to tmp_path
        tmp_path = self._sim_env.tmp_path
        table = self._sim_env.table
        hanger = self._sim_env.hanger
        garment = self._sim_env.garment
        grippers = self._sim_env.grippers
        renderer = self._sim_env.renderer

        table_path = os.path.join(tmp_path, "table.npy")
        hanger_path = os.path.join(tmp_path, "hanger.npy")
        garment_path = os.path.join(tmp_path, "garment.obj")
        garment_inv_path = os.path.join(tmp_path, "garment_inv.obj")
        robot_path = os.path.join(tmp_path, "robot.obj")
        gripper_l_path = os.path.join(tmp_path, "gripper_l.obj")
        gripper_r_path = os.path.join(tmp_path, "gripper_r.obj")

        table_pose = utils.torch_to_numpy(table.get_pos()[batch_idx])
        np.save(table_path, tra.translation_matrix(table_pose[0:3]) @ tra.quaternion_matrix(table_pose[3:7]))
        hanger_pose = utils.torch_to_numpy(hanger.get_pos()[batch_idx])
        np.save(hanger_path, tra.translation_matrix(hanger_pose[0:3]) @ tra.quaternion_matrix(hanger_pose[3:7]))
        garment.get_mesh(batch_idx, vert_norm=True).export(garment_path)
        invert_face(garment.get_mesh(batch_idx, vert_norm=True)).export(garment_inv_path)
        with open(robot_path, "w") as f_obj:
            json.dump(self._sim_env.get_robot_state_dict(batch_idx), f_obj, indent=4)
        if show_gripper:
            grippers["left"].get_mesh(batch_idx).export(gripper_l_path)
            grippers["right"].get_mesh(batch_idx).export(gripper_r_path)
        else:
            grippers["left"].get_mesh(batch_idx).apply_transform(tra.translation_matrix([0., 0., 5.])).export(gripper_l_path)
            grippers["right"].get_mesh(batch_idx).apply_transform(tra.translation_matrix([0., 0., 5.])).export(gripper_r_path)
        
        # render
        mask_str_to_idx = renderer.set_scene(
            camera_prop, sapien.Pose(camera_pose[0:3], camera_pose[3:7]),
            garment_filename=garment_path,
            garment_inv_filename=garment_inv_path,
            table_filename=table_path,
            hanger_filename=hanger_path,
            robot_filename=robot_path,
            gripper_l_filename=gripper_l_path,
            gripper_r_filename=gripper_r_path,
        )
        logger.info(f"current mask_id:{mask_str_to_idx}")
        
        return renderer.render(), mask_str_to_idx
    
    def e2e_step(self, action: torch.Tensor, steps_in_sim: int, callbacks: Optional[Callable]=None):
        assert action.shape == (self._sim_env.batch_size, ROBOT_DIM)

        if callbacks is None:
            callbacks = []
        
        # move joints
        target_qpos = {}
        target_qvel = {}
        for i, joint_name in enumerate(self._cfg.robot.joints):
            target_qpos[joint_name] = self._sim_env.robot.urdf.cfg[joint_name] + action[:, i + 2]
            limit = self._sim_env.robot.urdf.joint_map[joint_name].limit
            if limit is not None:
                target_qpos[joint_name] = torch.clamp(
                    target_qpos[joint_name], min(limit.lower, limit.upper), max(limit.lower, limit.upper), 
                )
            target_qvel[joint_name] = torch.zeros_like(target_qpos[joint_name])

        # fix qpos
        target_qpos = fix_joint_cfg(
            self._sim_env.robot.urdf, target_qpos, self._sim_env.get_table_height() + self._min_h,
            self.dtype, self.device, self.batch_size, self._hanger_transform,
        )

        # when use left gripper to grasp hanger, modify robot joints directly
        full_qpos = self._sim_env.robot.get_cfg_pos()
        full_qvel = self._sim_env.robot.get_cfg_vel()
        modified_qpos = move_left_to_hanger(
            self._sim_env.robot.urdf, full_qpos, 
            self.dtype, self.device, self.batch_size,
            so3.pos7d_to_matrix(self._sim_env.hanger.get_pos()), self._hanger_transform,
            action[:, :2], self.e2e_get_state()[:, :2],
        )
        for joint_name, joint_cfg in full_qpos.items():
            joint_cfg[...] = modified_qpos[joint_name]
        self._sim_env.robot.forward_kinematics(full_qpos, full_qvel)

        # set target
        self._sim_env.actor.set_actor_dof_targets(target_pos=target_qpos, target_vel=target_qvel)
        self._sim_env.set_actor_speed("interp", steps=steps_in_sim)

        # move grippers
        def garment_gripper_action(hand: str, batch_idx: int, action: Literal["Release", "Grasp"]):
            garment_gripper = self._sim_env.grippers[hand]
            if action == "Release":
                garment_gripper.set_mode_set_idx("Release", batch_idx)
            elif action == "Grasp":
                garment_gripper.set_mode_set_idx("Pick", batch_idx)
            else: raise ValueError(action)
            
        def robot_hanger_action(hand: str, batch_idx: int, action: Literal["Release", "Grasp"]):
            robot_hanger = self._sim_env.robot_hanger
            link_str = dict(
                left=self._agent.left_grasp_link,
                right=self._agent.right_grasp_link,
            )[hand]
            if action == "Release":
                if robot_hanger._link_str[batch_idx] == link_str:
                    robot_hanger.set_mode_set_idx("Release", batch_idx)
            elif action == "Grasp":
                robot_hanger.set_mode_set_idx("Fix", batch_idx, link_str=link_str, transform=self._hanger_transform[hand])
            else: raise ValueError(action)
            logger.info(f"{hand} {batch_idx} {action} {robot_hanger._current_mode}")

        def set_pick_to_hold(hand: str):
            garment_gripper = self._sim_env.grippers[hand]
            garment_gripper.callback(self._sim_env.env, self._sim_env.sim, substep=-1)
            idx = torch.where(garment_gripper._current_mode == garment_gripper.ModePick)
            garment_gripper._current_mode[idx] = garment_gripper.ModeHold
        
        gripper_action = utils.torch_to_numpy(action[:, :2])
        for hand_idx, hand in enumerate(["left", "right"]):
            for batch_idx in range(self.batch_size):
                if np.allclose(gripper_action[batch_idx, hand_idx], GRASP_NONE):
                    garment_gripper_action(hand, batch_idx, "Release")
                    robot_hanger_action(hand, batch_idx, "Release")
                elif np.allclose(gripper_action[batch_idx, hand_idx], GRASP_CLOTHES):
                    garment_gripper_action(hand, batch_idx, "Grasp")
                    robot_hanger_action(hand, batch_idx, "Release")
                elif np.allclose(gripper_action[batch_idx, hand_idx], GRASP_HANGER):
                    garment_gripper_action(hand, batch_idx, "Release")
                    robot_hanger_action(hand, batch_idx, "Grasp")
                else:
                    raise ValueError((hand_idx, action))
            set_pick_to_hold(hand)
        
        self._sim_env.simulate(steps_in_sim, callbacks=callbacks)

    def e2e_get_state(self):
        state = torch.zeros((self.batch_size, ROBOT_DIM), device=self.device, dtype=self.dtype)
        for i, joint_name in enumerate(self._cfg.robot.joints):
            state[:, i + 2] = self._sim_env.robot.urdf.cfg[joint_name]

        for batch_idx in range(self.batch_size):
            for hand_idx, hand in zip([0, 1], ["left", "right"]):
                gripper = self._sim_env.grippers[hand]
                # grasp garment
                if gripper._current_mode[batch_idx] != gripper.ModeRelease:
                    state[batch_idx, hand_idx] = GRASP_CLOTHES
                else:
                    robot_hanger = self._sim_env._robot_hanger
                    link_str = dict(
                        left=self._agent.left_grasp_link,
                        right=self._agent.right_grasp_link,
                    )[hand]
                    # grasp hanger
                    if (robot_hanger._current_mode[batch_idx] != robot_hanger.ModeRelease) and (robot_hanger._link_str[batch_idx] == link_str):
                        state[batch_idx, hand_idx] = GRASP_HANGER
        
        return state

    def e2e_get_action(self, state_prev: torch.Tensor, state_curr: torch.Tensor):
        action = state_curr.clone()
        for i, joint_name in enumerate(self._cfg.robot.joints):
            action[:, i + 2] -= state_prev[:, i + 2]
        return action
        
    def e2e_get_obs(self):
        """
        return depth_all, mask_garment_all, mask_hanger_all, [B, H, W]
        """
        color_all: List[np.ndarray] = []
        depth_all: List[np.ndarray] = []
        mask_garment_all: List[np.ndarray] = []
        mask_hanger_all: List[np.ndarray] = []

        torso_mat = self._sim_env.robot.urdf.link_transform_map["torso_base_link"]
        tf_mat = torch.tensor(np.array(self._cfg.camera.camera_extrinsics_in_torso_frame), dtype=self.dtype, device=self.device)
        for batch_idx in range(self._sim_env.batch_size):
            camera_mat = utils.torch_to_numpy(torso_mat[batch_idx] @ tf_mat)
            camera_pos = np.concatenate([camera_mat[0:3, 3], tra.quaternion_from_matrix(camera_mat)])

            result, mask_str_to_idx = self._take_picture(
                camera_intrinsic_to_property(np.array(self._cfg.camera.camera_intrinsics), self._cfg.camera.h, self._cfg.camera.w),
                camera_pos, batch_idx, show_gripper=False
            )
            color = result["rgba"]
            depth = result["depth"]
            mask_garment = (result["mask"] == mask_str_to_idx["garment"]).astype(np.int32)
            mask_hanger = (result["mask"] == mask_str_to_idx["hanger"]).astype(np.int32)

            color_all.append(color)
            depth_all.append(depth)
            mask_garment_all.append(mask_garment)
            mask_hanger_all.append(mask_hanger)
        
        return color_all, depth_all, mask_garment_all, mask_hanger_all


class ExporterE2E:
    def __init__(self, insert_gym: InsertGymE2E):
        self.insert_gym = insert_gym
        self.step_idx = 0
        self.traj_idx = 0

        self.inner_cnt = 0
        self.inner_num = 2 # export obs every 2 step
    
    def export(self):
        # export obs
        color_all, depth_all, mask_garment_all, mask_hanger_all = self.insert_gym.e2e_get_obs()

        for batch_idx in range(self.insert_gym.batch_size):
            obs_dir = os.path.join("e2e", f"{self.traj_idx}", f"{batch_idx}", "obs")
            for sub_dir in ["color", "depth", "mask_garment", "mask_hanger"]:
                os.makedirs(os.path.join(obs_dir, sub_dir), exist_ok=True)
            
            Image.fromarray(color_all[batch_idx]).save(os.path.join(obs_dir, "color", f"{str(self.step_idx).zfill(6)}.png"))
            np.save(os.path.join(obs_dir, "depth", f"{self.step_idx}.npy"), depth_all[batch_idx].astype(np.float16))
            np.save(os.path.join(obs_dir, "mask_garment", f"{self.step_idx}.npy"), mask_garment_all[batch_idx].astype(np.uint8))
            np.save(os.path.join(obs_dir, "mask_hanger", f"{self.step_idx}.npy"), mask_hanger_all[batch_idx].astype(np.uint8))
        
        # export state
        state_curr = self.insert_gym.e2e_get_state()
        for batch_idx in range(self.insert_gym.batch_size):
            state_path = os.path.join("e2e", f"{self.traj_idx}", f"{batch_idx}", "state", f"{self.step_idx}.npy")
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            np.save(state_path, utils.torch_to_numpy(state_curr[batch_idx, ...]))

        # export score
        json_file_path = os.path.join("e2e", f"{self.traj_idx}", "score", f"{self.step_idx}.json")
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, "w") as f_obj:
            json.dump(
                dict(
                    hanger_pos=utils.torch_to_numpy(self.insert_gym.sim_env.hanger.get_pos()).tolist(),
                    score=self.insert_gym.get_score(),
                    sim_error=self.insert_gym.sim_env.get_sim_error(), 
                ), fp=f_obj, indent=4,
            )

        # export prev step action
        if self.step_idx > 0:
            for batch_idx in range(self.insert_gym.batch_size):
                state_prev = np.load(os.path.join("e2e", f"{self.traj_idx}", f"{batch_idx}", "state", f"{self.step_idx - 1}.npy"))
                action = self.insert_gym.e2e_get_action(
                    torch.tensor(state_prev, dtype=self.insert_gym.dtype, device=self.insert_gym.device)[None, ...], 
                    state_curr[[batch_idx], ...]
                )
                action_path = os.path.join("e2e", f"{self.traj_idx}", f"{batch_idx}", "action", f"{self.step_idx - 1}.npy")
                os.makedirs(os.path.dirname(action_path), exist_ok=True)
                np.save(action_path, utils.torch_to_numpy(action[0, ...]))
        
        self.step_idx += 1

    def callback(self, env: Env, sim: Sim, substep: int):
        if substep == sim.substeps - 1: # at the end of a step
            self.inner_cnt += 1
            if self.inner_cnt == self.inner_num:
                self.export()
                self.inner_cnt = 0
    
    def update_traj_idx(self):
        json_file_path = os.path.join("e2e", f"{self.traj_idx}", "e2e_info.json")
        os.makedirs(os.path.dirname(json_file_path), exist_ok=True)
        with open(json_file_path, "w") as f_obj:
            json.dump(
                dict(
                    batch_size=self.insert_gym.batch_size,
                    step_cnt=self.step_idx,
                    score=self.insert_gym.get_score(), 
                    sim_error=self.insert_gym.sim_env.get_sim_error(), 
                ), fp=f_obj, indent=4,
            )
        self.step_idx = 0
        self.traj_idx += 1
