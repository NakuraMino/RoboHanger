import logging
logger = logging.getLogger(__name__)

import sys
import shutil
import copy
from typing import Dict, Literal, Optional, Union, Any, List
import pprint
import time as time_module
import math
from dataclasses import dataclass
import os
import threading
import json

import trimesh
import numpy as np
import torch
import torch.nn.functional as F
import trimesh.transformations as tra
import matplotlib.pyplot as plt
from PIL import Image

import omegaconf
from pdf2image import convert_from_path

import batch_urdf
import robohang.common.utils as utils
import robohang.sim.so3 as so3
from robohang.policy.learn_utils import plot_wrap_fig
from robohang.policy.policy_utils import MetaInfo
from robohang.policy.funnel.funnel_learn import FunnelActEncDec
from robohang.policy.funnel.funnel_gym import clip_fling_action
from robohang.real.realapi import RealAPI, CONTROLLABLE_JOINT_LIST
from robohang.real.policy import (
    RealPolicy,
    depth_to_point_cloud,
    align_point_cloud,
    sample_point_cloud,
    e2e_fix_depth,
)
from robohang.real.sapien_visualizer import SapienVisualize, HangerVisualizerMode
import robohang.real.utils as real_utils
import robohang.real.sam_util as sam_util
from robohang.env.sapien_renderer import (
    camera_property_to_intrinsics_matrix,
    camera_pose_to_matrix,
    CameraProperty,
    xyz2uv,
)
from robohang.policy.insert.insert_gym_e2e import (
    GRASP_NONE, GRASP_CLOTHES, GRASP_HANGER, fix_joint_cfg, 
)
from robohang.policy.insert.insert_learn_act import MAX_DEPTH


HANGER_XY_CACHE_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cache", "policy", "cache.json")
os.makedirs(os.path.dirname(HANGER_XY_CACHE_FILE), exist_ok=True)
FINAL_SHOW = False


def overwrite_insert_action_space_cfg(
    press_action_space: omegaconf.DictConfig,
    lift_action_space: omegaconf.DictConfig,
    drag_action_space: omegaconf.DictConfig,
    rotate_action_space: omegaconf.DictConfig,
):
    MetaInfo.press_action_space = press_action_space
    MetaInfo.lift_action_space = lift_action_space
    MetaInfo.drag_action_space = drag_action_space
    MetaInfo.rotate_action_space = rotate_action_space



class E2EFixQpos:
    def __init__(
        self, dtype, device, urdf_path: str, mesh_dir: str,
        global_offset: omegaconf.DictConfig,
        robot_cfg: omegaconf.DictConfig,
        robot_ik_solver_cfg: omegaconf.DictConfig,
    ) -> None:
        # create urdf object
        self._dtype, self._device = dtype, device
        self._global_offset = copy.deepcopy(global_offset)
        self._robot_cfg = copy.deepcopy(robot_cfg)
        self._robot_ik_solver_cfg = copy.deepcopy(robot_ik_solver_cfg)
        
        self._urdf = batch_urdf.URDF(
            batch_size=1,
            urdf_path=utils.get_path_handler()(urdf_path),
            dtype=self._dtype,
            device=self._device,
            mesh_dir=utils.get_path_handler()(mesh_dir),
        )
    
    def _to_full_cfg(self, cfg: Optional[Dict[str, torch.Tensor]]):
        if cfg is None:
            return None
        full_cfg = {k: v.clone() for k, v in self._urdf.cfg.items()}
        for k, v in cfg.items():
            full_cfg[k][...] = v
        return full_cfg

    def _tensor(self, x, dtype=None):
        if dtype is None:
            dtype = self._dtype
        return torch.tensor(np.array(x), dtype=dtype, device=self._device)
    
    def add_ee_offset_to_qpos(self, qpos: Dict[str, float]):
        qpos_full = self._to_full_cfg(qpos)
        self._urdf.update_cfg(qpos_full)
        
        mask = torch.ones(16, dtype=self._dtype, device=self._device)
        def add_offset_to_single_hand(hand_name: str):
            mat = self._urdf.link_transform_map[
                getattr(self._robot_cfg.grasp, hand_name).link
            ].clone() # [B, 4, 4]
            mat[:, :3, 3] += self._tensor([getattr(self._global_offset, hand_name).xyz])
            def err_func(link_transform_map: Dict[str, torch.Tensor]):
                curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link].view(1, 16, 16) # [B, 16, 16]
                err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(1, 16) # [B, 16]
                return err_mat * mask
            def loss_func(link_transform_map: Dict[str, torch.Tensor]):
                curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link]
                err_mat = curr_mat4 - mat # [B, 16]
                return torch.sum(torch.square(err_mat).view(1, 16) * mask, dim=1) # [B, ]
            if hand_name in ["left", "right"]:
                new_cfg, info = self._urdf.inverse_kinematics_optimize(
                    err_func=err_func,
                    loss_func=loss_func,
                    **(self._robot_ik_solver_cfg),
                )
                logger.info(f"add_ee_offset_to_qpos {hand_name}")
                logger.info(pprint.pformat(new_cfg))
                for joint_name in omegaconf.OmegaConf.to_container(
                    getattr(self._robot_cfg.grasp, hand_name).joints
                ):
                    qpos_full[joint_name][...] = new_cfg[joint_name]
            else:
                raise NotImplementedError(hand_name)
        
        add_offset_to_single_hand("left")
        add_offset_to_single_hand("right")
        return qpos_full


class RealWorld:
    def __init__(self, real_api: RealAPI, cfg: omegaconf.DictConfig, policy_cfg: omegaconf.DictConfig, global_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(real_api, RealAPI)
        assert isinstance(cfg, omegaconf.DictConfig)
        assert isinstance(global_cfg, omegaconf.DictConfig)

        self._cfg = copy.deepcopy(cfg)
        
        overwrite_insert_action_space_cfg(
            self._cfg.primitive.insert.press.action_space,
            self._cfg.primitive.insert.lift.action_space,
            self._cfg.primitive.insert.drag.action_space,
            self._cfg.primitive.insert.rotate.action_space,
        )
        
        # real api
        self._real_api = real_api
        self._qpos_cache = []
        self._time_cache = []

        # factory args
        self._dtype: torch.dtype = getattr(torch, global_cfg.default_float)
        assert isinstance(self._dtype, torch.dtype)
        self._dtype_int: torch.dtype = getattr(torch, global_cfg.default_int)
        assert isinstance(self._dtype_int, torch.dtype)
        self._device: str = str(global_cfg.torch_device)

        # object
        self._init_urdf()
        self._init_table()
        self._init_hanger()
        
        # policy
        self._policy = RealPolicy(policy_cfg, global_cfg)
        self._policy.set_funnel_action_space(self._cfg.primitive.fling.action_space)
        self._inference_cnt: Dict[Literal["f", "il", "ir", "kp"], int] = dict(
            f=0, il=0, ir=0, kp=0,
        )
        self._reproject_delta_depth = float(self._cfg.obs.reproject.delta_depth)
        self._policy_cache = dict()
        if os.path.exists(HANGER_XY_CACHE_FILE):
            with open(HANGER_XY_CACHE_FILE, "r") as f_obj:
                self._policy_cache["hanger_xy"] = np.array(json.load(f_obj)["hanger_xy"])
        
        # end to end policy
        self._policy_e2e_cache = dict(left_gripper=0., right_gripper=0.)
        self._fix_qpos = E2EFixQpos(
            self._dtype, self._device, self._robot_cfg.urdf_path, self._hanger_cfg.mesh_path,
            self._cfg.global_var.offset_e2e, self._robot_cfg, self._robot_ik_solver_cfg,
        )
        
        # visualizer
        self._use_visualize = bool(self._cfg.utils.visualize.use)
        if self._use_visualize:
            self._visualizer = SapienVisualize(
                viewer_cfg=self._cfg.utils.visualize.viewer,
                urdf_path=utils.get_path_handler()(self._robot_cfg.urdf_path),
                hanger_path=utils.get_path_handler()(self._hanger_cfg.mesh_path),
                table_height=self._table_height,
                device=self._device,
            )
        real_utils.vis.enable(bool(self._cfg.utils.show_window))
        
        # save camera image
        # self._launch_camera_thread()
        
        # primitive parameters
        self._primitive_param: omegaconf.DictConfig = copy.deepcopy(cfg.primitive.parameter)
        
    def _tensor(self, x, dtype=None):
        if dtype is None:
            dtype = self._dtype
        return torch.tensor(np.array(x), dtype=dtype, device=self._device)
    
    def _init_urdf(self):
        self._robot_cfg = self._cfg.obj.robot

        # create urdf object
        self._urdf = batch_urdf.URDF(
            batch_size=1,
            urdf_path=utils.get_path_handler()(self._robot_cfg.urdf_path),
            dtype=self._dtype,
            device=self._device,
            mesh_dir=utils.get_path_handler()(self._robot_cfg.mesh_dir),
        )

        # init qpos
        self._init_qpos = {k: v.clone() for k, v in self._urdf.cfg.items()}
        for k, v in self._robot_cfg.init.qpos.items():
            self._init_qpos[k][...] = v
        self._urdf.update_cfg(self._init_qpos)
        self._robot_target_qpos = {k: v.clone() for k, v in self._urdf.cfg.items()}
        """cache target qpos with batch_size = 1"""

        # init root transformation
        self._robot_root_mat = (
            tra.translation_matrix(self._robot_cfg.init.base.pos[:3]) @
            tra.quaternion_matrix(self._robot_cfg.init.base.pos[3:])
        )
        self._urdf.update_base_link_transformation(
            self._robot_cfg.init.base.name, 
            self._tensor(self._robot_root_mat)[None, ...]
        )

        # inverse kinematic
        self._robot_ik_init = copy.deepcopy(self._robot_cfg.ik.init)
        self._robot_ik_solver_cfg = copy.deepcopy(self._robot_cfg.ik.solver)
        
    def _init_table(self):
        table_cfg = self._cfg.obj.table
        self._table_height = float(table_cfg.height)
    
    def _init_hanger(self):
        self._hanger_cfg = self._cfg.obj.hanger

        hanger_path = utils.get_path_handler()(self._hanger_cfg.mesh_path)
        self._hanger_mesh: trimesh.Trimesh = trimesh.load_mesh(hanger_path)
        self._hanger_meta = omegaconf.OmegaConf.load(hanger_path + ".meta.yaml")
        self._hanger_translate_xy: np.ndarray = np.array(self._hanger_cfg.translate_xy)
        
        self._hanger_visualizer_mode = HangerVisualizerMode("none", np.eye(4))
        self._hanger_surface = self._tensor(self._hanger_mesh.sample(self._hanger_cfg.sample_num))
        """float, [S, 3]"""
        self._hanger_surface[:, :2] -= self._tensor(self._hanger_translate_xy)

    def _get_hanger_endpoint_xyz_meta(self, endpoint: str):
        xyz = np.array(getattr(self._hanger_meta, endpoint))
        return xyz * 1.0 # for safety margin
    
    def _get_head_camera_extrinsics(self) -> np.ndarray:
        e = self._real_api.get_head_camera_extrinsics()
        qpos = self._urdf.cfg
        qpos["head_joint1"][...] = self._real_api.head_joints[0]
        qpos["head_joint2"][...] = self._real_api.head_joints[1]
        self._urdf.update_cfg(qpos)
        ans = utils.torch_to_numpy(self._urdf.link_transform_map[e.from_link])[0, :, :] @ (
            tra.translation_matrix(e.translation) @ 
            tra.quaternion_matrix(np.array(e.rotation_xyzw)[[3, 0, 1, 2]]) @ 
            np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]]) # model matrix
        )
        print(f"camera_extrinsic:{tra.euler_from_matrix(ans)}\n{ans}")
        return ans
    
    def _get_head_camera_intrinsics(self) -> np.ndarray:
        ans = self._real_api.get_head_camera_intrinsics()
        return ans

    def _get_real_reproject_camera_z(self):
        return float(self._cfg.obs.reproject.camera_pose[2])
    
    def _get_neckline_prompt(self, intrinsics_matrix, extrinsics_matrix, seg_neckline_overwrite=None, y_offset=0.):
        input_point = []
        input_label = []
        for i, xyz in enumerate(self._cfg.obs.neckline_prompt.pos):
            xyz = np.array(xyz)
            if i == 0 and seg_neckline_overwrite is not None:
                xyz[1] = float(seg_neckline_overwrite) # only for test
            xyz[1] += y_offset
            input_point.append(xyz2uv(np.array(xyz), extrinsics_matrix, intrinsics_matrix))
            input_label.append(1)
        for xyz in self._cfg.obs.neckline_prompt.neg:
            input_point.append(xyz2uv(np.array(xyz), extrinsics_matrix, intrinsics_matrix))
            input_label.append(0)
        return dict(
            input_point=np.array(input_point),
            input_label=np.array(input_label),
        )
    
    def _get_mask_neckline(self, img: np.ndarray, seg_neckline_overwrite=None):
        intrinsics_matrix = self._get_head_camera_intrinsics()
        extrinsics_matrix = self._get_head_camera_extrinsics()
        
        prompt = self._get_neckline_prompt(intrinsics_matrix, extrinsics_matrix, seg_neckline_overwrite=seg_neckline_overwrite)
        input_point = prompt["input_point"]
        input_label = prompt["input_label"]
        mask = self._real_api.get_mask_use_sam(img, input_point, input_label)[0]
        if self._cfg.obs.neckline_prompt.visualize_prompt:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(img)
            sam_util.show_mask(mask, plt.gca())
            sam_util.show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            fig.canvas.draw()
            vis_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            real_utils.vis.show(vis_img)
            plt.close()
        return mask
    
    def _get_mask_neckline_best(self, img: np.ndarray, clothes_mask: np.ndarray):
        intrinsics_matrix = self._get_head_camera_intrinsics()
        extrinsics_matrix = self._get_head_camera_extrinsics()
        
        prompt_cfg = self._cfg.obs.neckline_prompt
        best_idx, best_mask = None, None
        for y_offset_idx, y_offset in enumerate(prompt_cfg.y_offset_list):
            prompt = self._get_neckline_prompt(intrinsics_matrix, extrinsics_matrix, y_offset=y_offset)
            input_point = prompt["input_point"]
            input_label = prompt["input_label"]
            mask = self._real_api.get_mask_use_sam(img, input_point, input_label)[0]
            mask_sum = mask.sum()
            logger.info(f"mask_sum:{mask_sum} shape:{mask.shape}")
            if best_idx is None:
                best_idx, best_mask = y_offset_idx, mask
            if (
                prompt_cfg.desired_pixel_cnt_min < mask_sum < prompt_cfg.desired_pixel_cnt_max and
                np.logical_and(mask > 0.5, clothes_mask > 0.5).sum() > mask_sum * 0.9
            ):
                best_idx, best_mask = y_offset_idx, mask
                break
        logger.info(f"mask neckline use the {best_idx} th y_offset")
        
        if prompt_cfg.visualize_prompt:
            fig = plt.figure(figsize=(10, 10))
            plt.imshow(img)
            sam_util.show_mask(best_mask, plt.gca())
            sam_util.show_points(input_point, input_label, plt.gca())
            plt.axis('off')
            fig.canvas.draw()
            vis_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            real_utils.vis.show(vis_img)
            plt.close()
        return best_mask
        
    def _get_processed_obs(
        self, seg_neckline=False, move_to_obs_leg_joint=False, seg_neckline_overwrite: Optional[float]=None,
    ) -> Dict[Literal["depth", "clothes", "hanger", "neckline", "raw_rgb"], np.ndarray]:
        if move_to_obs_leg_joint:
            qpos = {k: float(v) for k, v in self._init_qpos.items()}
            for k, v in qpos.items():
                if "leg_joint" in k:
                    qpos[k] = getattr(self._cfg.obs.obs_leg_joint, k)
            self._api_move_to_qpos(qpos, cache=False)
        
        reproject_cfg = self._cfg.obs.reproject
        obs_raw = self._real_api.get_obs()
        title_list = ["depth", "clothes", "hanger", "rgb"]
        if seg_neckline:
            if seg_neckline_overwrite is None:
                mask_neckline = self._get_mask_neckline_best(obs_raw["rgb"], obs_raw["clothes"])
            else:
                mask_neckline = self._get_mask_neckline(obs_raw["rgb"], seg_neckline_overwrite)
            fig = plot_wrap_fig(
                [obs_raw[k][None, ...] for k in title_list] + [mask_neckline[None, ...]], 
                title_list + ["neckline"], ["jet", "gray", "gray", None, "gray"], 
                1, width_unit=6.
            )
            fig.canvas.draw()
            vis_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
            real_utils.vis.show(vis_img)
            plt.close()
        else:
            mask_neckline = obs_raw["clothes"].copy()

        def call_reproject(depth, mask):
            reproject_result = real_utils.reproject(
                depth_input=depth, mask_input=mask,
                output_shape=(reproject_cfg.camera_prop.height, reproject_cfg.camera_prop.width),
                intrinsics_matrix_input=self._get_head_camera_intrinsics(),
                intrinsics_matrix_output=camera_property_to_intrinsics_matrix(CameraProperty(**(reproject_cfg.camera_prop))),
                camera_pose_input=self._get_head_camera_extrinsics(),
                camera_pose_output=camera_pose_to_matrix(reproject_cfg.camera_pose),
                interp_mask=False, x1x2y1y2=np.array([-1e6, 1e6, -1e6, reproject_cfg.input_max_y]),
            )
            return reproject_result["depth_output"], reproject_result["mask_output"]
        
        depth, clothes = call_reproject(obs_raw["depth"].astype(np.float32), obs_raw["clothes"].astype(np.int32))
        _, hanger = call_reproject(obs_raw["depth"].astype(np.float32), obs_raw["hanger"].astype(np.int32))
        _, neckline = call_reproject(obs_raw["depth"].astype(np.float32), mask_neckline.astype(np.int32))
        
        # soft clip depth
        # max_depth = reproject_cfg.max_depth
        # print("max_depth", max_depth)
        # depth = np.clip(depth, 0., max_depth)
        # depth = np.where(depth < max_depth, depth, max_depth + (1. - np.exp(max_depth - depth)) * 0.01)

        if self._cfg.obs.overwrite_reproject_depth is not None:
            logger.warn("overwrite obs !")
            depth[...] = float(self._cfg.obs.overwrite_reproject_depth) # for test
            depth += np.random.randn(*(depth.shape)) * 0.01 # for test

        if move_to_obs_leg_joint:
            self._api_move_to_qpos(self._init_qpos, cache=False)
        return dict(
            depth=depth,
            clothes=clothes,
            hanger=hanger,
            neckline=neckline,
            raw_rgb=obs_raw["rgb"],
        )
    
    def _get_hanger_point_cloud(self):
        obs_raw = self._real_api.get_obs()
        pc_full = depth_to_point_cloud(
            obs_raw["depth"], 
            self._get_head_camera_extrinsics(),
            self._get_head_camera_intrinsics(),
        )
        trimesh.PointCloud(pc_full).export("full.ply")
        pc_hanger = pc_full[np.where(obs_raw["hanger"].reshape(-1))[0], :]
        trimesh.PointCloud(pc_hanger).export("hanger.ply")
        shutil.copy("rgb_image.png", "hanger.png")
        return pc_hanger
    
    def _to_full_cfg(self, cfg: Optional[Dict[str, torch.Tensor]]):
        if cfg is None:
            return None
        full_cfg = {k: v.clone() for k, v in self._urdf.cfg.items()}
        for k, v in cfg.items():
            full_cfg[k][...] = v
        return full_cfg

    def _visualizer_set_hanger_mode(self, hand: Literal["none", "left", "right"], origin: Optional[np.ndarray]=None):
        if hand == "none":
            self._hanger_visualizer_mode = HangerVisualizerMode("none", np.eye(4))
        elif hand in ["left", "right"]:
            assert isinstance(origin, np.ndarray)
            assert origin.shape == (4, 4), origin.shape
            self._hanger_visualizer_mode = HangerVisualizerMode(self._get_end_effector_name(hand), origin)
        else:
            raise ValueError(hand)
    
    def _visualizer_run(self, curr_qpos: Dict[str, float], targ_qpos: Dict[str, float], time: Optional[float]=None):
        if time is None:
            time = self._cfg.utils.visualize.default_time
        time = float(time)
        steps = int(time / self._visualizer.timestep) + 1
        for s in range(1, steps + 1):
            self._visualizer.step(
                cfg = {
                    k: (curr_qpos[k] * (steps - s) + targ_qpos[k] * s) / steps for k in curr_qpos.keys()
                }, 
                root_mat=self._robot_root_mat,
                hanger_mode=self._hanger_visualizer_mode
            )
            self._visualizer.render()
    
    def _api_move_to_qpos(self, qpos: Dict[str, Any], time: Optional[float]=None, cache=False, overwrite_single_step=None, qpos_to_exec=None):
        # update cfg in targ qpos
        curr_qpos = {k: float(v) for k, v in self._urdf.cfg.items()}
        targ_qpos = {k: float(v) for k, v in qpos.items()}
        self._urdf.update_cfg(self._to_full_cfg(targ_qpos))
        for k, v in self._urdf.cfg.items():
            self._robot_target_qpos[k][...] = v

        # visualize
        if self._use_visualize:
            self._visualizer_run(curr_qpos, targ_qpos, time)

        # real api
        if not cache:
            assert len(self._qpos_cache) == 0, "please run self._api_run_all_cache() first"
            exec_targ_qpos = targ_qpos if qpos_to_exec is None else {k: float(v) for k, v in qpos_to_exec.items()}
            self._real_api.move_to_qpos({k: float(exec_targ_qpos[k]) for k in CONTROLLABLE_JOINT_LIST}, time, overwrite_single_step=overwrite_single_step)
        else:
            if len(self._qpos_cache) == 0:
                self._qpos_cache.append({k: float(curr_qpos[k]) for k in CONTROLLABLE_JOINT_LIST}) # at least 2 qpos in the qpos cache
                self._time_cache.append(0.)
            self._qpos_cache.append({k: float(targ_qpos[k]) for k in CONTROLLABLE_JOINT_LIST})
            self._time_cache.append(float(time))
    
    def _api_manipulate_gripper(
        self, 
        left: Literal["none", "close", "open"],
        right: Literal["none", "close", "open"],
        overwrite_single_step=None,
    ):
        # update cfg in targ qpos
        curr_qpos = {k: float(v) for k, v in self._urdf.cfg.items()}
        targ_qpos = {k: float(v) for k, v in self._urdf.cfg.items()}
        grasp_cfg = self._robot_cfg.grasp
        for handname, action in zip(["left", "right"], [left, right]):
            if action in ["close", "open"]:
                joints_name_dict: dict = omegaconf.OmegaConf.to_container(getattr(getattr(grasp_cfg, handname), action))
                for j, v in joints_name_dict.items():
                    targ_qpos[j] = v
        self._urdf.update_cfg(self._to_full_cfg(targ_qpos))
        for k, v in self._urdf.cfg.items():
            self._robot_target_qpos[k][...] = v

        # visualize
        if self._use_visualize:
            self._visualizer_run(curr_qpos, targ_qpos)

        # real api
        self._real_api.control_gripper(left=left, right=right, overwrite_single_step=overwrite_single_step)
    
    def _api_run_all_cache(self):
        joint_limit = {k: [v.limit.lower, v.limit.upper] if v.limit is not None else None for k, v in self._urdf.joint_map.items()}
        self._real_api.move_to_qpos_list(self._qpos_cache, joint_limit, self._time_cache)
        self._qpos_cache = []
        self._time_cache = []

    def _move_to_robot_target_qpos(self, time: Optional[float]=None, cache=False):
        self._api_move_to_qpos(qpos=self._robot_target_qpos, time=time, cache=cache)
    
    def _calculate_xyz_c(self, xyz_l: Optional[torch.Tensor], xyz_r: Optional[torch.Tensor]):
        if (xyz_l is not None) and (xyz_r is not None):
            return (xyz_l + xyz_r) / 2
        elif (xyz_l is not None):
            return xyz_l.clone()
        elif (xyz_r is not None):
            return xyz_r.clone()
        else:
            raise ValueError("xyz_l, xyz_r are both None")
    
    def _endpoint_xyz_to_hanger_xyz(
        self,
        endpoint_name: Literal["left", "right"],
        endpoint_xyz: torch.Tensor,
        hanger_rpy: np.ndarray,
    ):
        """
        endpoint_xyz: world frame endpoint's xyz
        hanger_rpy: world frame hanger's rpy
        """
        rot_mat = tra.euler_matrix(*hanger_rpy)[:3, :3]
        xyz_hanger_frame = self._get_hanger_endpoint_xyz_meta(endpoint_name)
        xyz_world_frame = rot_mat @ xyz_hanger_frame
        return endpoint_xyz - self._tensor(xyz_world_frame)
    
    def _hanger_xyz_to_endpoint_xyz(
        self,
        endpoint_name: Literal["left", "right"],
        hanger_xyz: torch.Tensor,
        hanger_rpy: np.ndarray,
    ):
        """
        hanger_xyz: world frame hanger's xyz
        hanger_rpy: world frame hanger's rpy
        """
        rot_mat = tra.euler_matrix(*hanger_rpy)[:3, :3]
        xyz_hanger_frame = self._get_hanger_endpoint_xyz_meta(endpoint_name)
        xyz_world_frame = rot_mat @ xyz_hanger_frame
        return hanger_xyz + self._tensor(xyz_world_frame)

    def _get_end_effector_name(self, hand: Literal["left", "right"]) -> str:
        return getattr(self._robot_cfg.grasp, hand).link
    
    def _get_end_effector_mat(self, hand: Literal["left", "right"]):
        return self._urdf.link_transform_map[self._get_end_effector_name(hand)]

    def _get_current_xyz(self, endpoint_name: Literal["left", "right"], hanger_mat: torch.Tensor):
        """return xyz_world_frame, [B, 3]"""
        xyz_hanger_frame = self._tensor(
            [*(self._get_hanger_endpoint_xyz_meta(endpoint_name)), 1.],
        )[None, :] # [B, 4]
        xyz_world_frame = (hanger_mat @ xyz_hanger_frame[..., None])[:, :3, 0] # [B, 3]
        return xyz_world_frame.contiguous()
    
    def _set_gripper_target_wrap(
        self,
        xyz: torch.Tensor,
        hand_name: str,
        use_ik_init_cfg: bool, 
        xyz_c: torch.Tensor,
        leg_joint_cfg: Optional[omegaconf.DictConfig]=None,
        rot: Optional[torch.Tensor]=None,
    ):
        assert xyz.shape == (1, 3), xyz.shape
        logger.info(f"set_gripper_target_wrap xyz:\n{xyz}")
        # target mat
        mat = torch.eye(4, dtype=self._dtype, device=self._device)[None, ...]
        mat[:, :3, 3] = xyz + self._tensor([getattr(self._cfg.global_var.offset, hand_name).xyz])
        if rot is not None:
            mat[:, :3, :3] = rot

        # init cfg
        if use_ik_init_cfg:
            init_cfg = {k: v for k, v in getattr(self._robot_ik_init, hand_name).items()}
        else:
            init_cfg = {}

        # set leg joints
        leg_joints = ["leg_joint3", "leg_joint4"]
        logger.info(f"leg_joint_cfg:\n{leg_joint_cfg}")
        if leg_joint_cfg is None:
            leg_joint_cfg = omegaconf.DictConfig({})
        for leg_joint in leg_joints:
            cfg = getattr(leg_joint_cfg, leg_joint, omegaconf.DictConfig({}))
            if hasattr(cfg, "val"):
                init_cfg[leg_joint] = float(cfg.val)
            elif hasattr(cfg, "coeff"):
                limit = self._urdf.joint_map[leg_joint].limit
                if leg_joint == "leg_joint3":
                    init_cfg[leg_joint] = (
                        self._init_qpos[leg_joint] + (xyz_c[:, 1] - 0.3).clamp(0., 0.4) * float(cfg.coeff)
                    ).clamp(limit.lower, limit.upper) # proportional to y value
                elif leg_joint == "leg_joint4":
                    init_cfg[leg_joint] = (
                        self._init_qpos[leg_joint] + xyz_c[:, 0].clamp(-0.2, 0.2) * float(cfg.coeff)
                    ).clamp(limit.lower, limit.upper) # proportional to x value
                else:
                    raise NotImplementedError(leg_joint)
            else:
                init_cfg[leg_joint] = self._init_qpos[leg_joint]

        # common ik
        if rot is None:
            mask = self._tensor([0., 0., 0., 1.] * 2 + [1., 0., 0., 1.] + [0., 0., 0., 1.]) # (e'_x)_z = 0
        else:
            mask = torch.ones(16, dtype=self._dtype, device=self._device)
        def err_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link].view(1, 16, 16) # [B, 16, 16]
            err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(1, 16) # [B, 16]
            return err_mat * mask
        def loss_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link]
            err_mat = curr_mat4 - mat # [B, 16]
            return torch.sum(torch.square(err_mat).view(1, 16) * mask, dim=1) # [B, ]
        if hand_name in ["left", "right"]:
            new_cfg, info = self._urdf.inverse_kinematics_optimize(
                err_func=err_func,
                loss_func=loss_func,
                init_cfg=self._to_full_cfg(init_cfg),
                **(self._robot_ik_solver_cfg),
            )
            logger.info("inverse kinematics")
            logger.info(pprint.pformat(info))
            for joint_name in omegaconf.OmegaConf.to_container(
                getattr(self._robot_cfg.grasp, hand_name).joints
            ) + leg_joints:
                self._robot_target_qpos[joint_name][...] = new_cfg[joint_name]
        else:
            raise NotImplementedError(hand_name)
    
    def _keypoints_clip_input(self, xy1: torch.Tensor, xy2: torch.Tensor, dist_range: List[float], xyc: List[float], radius_max: float):
        # process xym
        xym = (xy1 + xy2) / 2.
        xyc = self._tensor(xyc)
        xym_to_xyc = xym - xyc
        xym_to_xyc_len = xym_to_xyc.norm(dim=1)
        n = xym_to_xyc / xym_to_xyc_len[:, None]
        too_far_indices = torch.where(xym_to_xyc_len > radius_max)[0]
        xym_to_xyc_new = xym_to_xyc.clone()
        xym_to_xyc_new[too_far_indices, :] = n[too_far_indices, :] * radius_max
        xym_new = xym_to_xyc_new + xyc
        
        # process xyd
        xyd = (xy2 - xy1)
        eps = 1e-5
        xyd_len = xyd.norm(dim=1)
        n = xyd / xyd_len[:, None]
        n[torch.where(xyd_len < eps)[0], :] = self._tensor([0., 1.])
        xyd_new = n * torch.clamp(xyd_len, min(dist_range), max(dist_range))

        return xym_new - xyd_new / 2, xym_new + xyd_new / 2
        
    def _sequence_move_to_pick_points(
        self,
        xyz_l: Optional[torch.Tensor],
        xyz_r: Optional[torch.Tensor],
        pick_points_cfg: omegaconf.DictConfig,
        h_lower_l: float=0.,
        h_lower_r: float=0.,
        rot_l: Optional[torch.Tensor] = None,
        rot_r: Optional[torch.Tensor] = None,
    ):
        """
        z in xyz_l and xyz_r is the table height
        """ 
        logger.info(f"in sequence_move_to_pick_points(), h_lower_l:{h_lower_l} h_lower_r:{h_lower_r}")
        
        def type_check(x):
            if x is not None:
                assert isinstance(x, torch.Tensor)
                assert x.shape == (1, 3)
        type_check(xyz_l), type_check(xyz_r)

        xyz_c = self._calculate_xyz_c(xyz_l, xyz_r)

        logger.info("[STAGE] move to h_upper")
        if xyz_l is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_l + self._tensor([0., 0., pick_points_cfg.h_upper]), 
                rot=rot_l,
                hand_name="left", use_ik_init_cfg=True, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        if xyz_r is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_r + self._tensor([0., 0., pick_points_cfg.h_upper]), 
                rot=rot_r,
                hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        self._move_to_robot_target_qpos(pick_points_cfg.time[0])

        logger.info("[STAGE] move to h_lower")
        if xyz_l is not None:
            h_lower = pick_points_cfg.h_lower + h_lower_l
            self._set_gripper_target_wrap(
                xyz=xyz_l + self._tensor([0., 0., h_lower]), 
                rot=rot_l,
                hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        if xyz_r is not None:
            h_lower = pick_points_cfg.h_lower + h_lower_r
            self._set_gripper_target_wrap(
                xyz=xyz_r + self._tensor([0., 0., h_lower]), 
                rot=rot_r,
                hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        self._move_to_robot_target_qpos(pick_points_cfg.time[1])

        logger.info("[STAGE] close")
        self._api_manipulate_gripper(None if xyz_l is None else "close", None if xyz_r is None else "close")

        logger.info("[STAGE] move to h_later")
        if xyz_l is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_l + self._tensor([0., 0., pick_points_cfg.h_later]), 
                rot=rot_l,
                hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        if xyz_r is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_r + self._tensor([0., 0., pick_points_cfg.h_later]), 
                rot=rot_r,
                hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c, leg_joint_cfg=pick_points_cfg.leg
            )
        self._move_to_robot_target_qpos(pick_points_cfg.time[2])

    def _sequence_move_to_h(
        self, 
        cfg: omegaconf.DictConfig,
        xyz_l: torch.Tensor=None,
        xyz_r: torch.Tensor=None,
    ):
        xyz_c = self._calculate_xyz_c(xyz_l, xyz_r)
        if xyz_l is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_l + self._tensor(
                    cfg.left.xyh
                ), hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c, 
            )
        if xyz_r is not None:
            self._set_gripper_target_wrap(
                xyz=xyz_r + self._tensor(
                    cfg.right.xyh
                ), hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c, 
            )
        self._move_to_robot_target_qpos(cfg.time)
    
    def _sequence_liftup_flingforward(
        self,
        xyz_l: torch.Tensor,
        xyz_r: torch.Tensor,
        lift_up_cfg: omegaconf.DictConfig,
        fling_forwards_cfg: List[omegaconf.DictConfig],
    ):
        logger.info("[STAGE] lift up")
        dist = (xyz_l - xyz_r).norm(dim=-1)
        
        lift_l = torch.zeros((1, 3), dtype=self._dtype, device=self._device)
        lift_l[:, 0] = -dist * lift_up_cfg.x.scale / 2.
        lift_l[:, 1] = lift_up_cfg.y
        lift_l[:, 2] = lift_up_cfg.h + self._table_height
        
        lift_r = torch.zeros((1, 3), dtype=self._dtype, device=self._device)
        lift_r[:, 0] = +dist * lift_up_cfg.x.scale / 2.
        lift_r[:, 1] = lift_up_cfg.y
        lift_r[:, 2] = lift_up_cfg.h + self._table_height

        xyz_c = self._calculate_xyz_c(lift_l, lift_r)
        self._set_gripper_target_wrap(xyz=lift_l, hand_name="left", use_ik_init_cfg=True, xyz_c=xyz_c)
        self._set_gripper_target_wrap(xyz=lift_r, hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_c)
        self._move_to_robot_target_qpos(lift_up_cfg.time)

        logger.info("[STAGE] fling forward")
        for fling_forward_cfg in fling_forwards_cfg:
            fling_l = torch.zeros((1, 3), dtype=self._dtype, device=self._device)
            fling_l[:, 0] = lift_l[:, 0] * getattr(fling_forward_cfg, "x_scale", 1.)
            fling_l[:, 1] = fling_forward_cfg.y
            fling_l[:, 2] = fling_forward_cfg.h + self._table_height

            fling_r = torch.zeros((1, 3), dtype=self._dtype, device=self._device)
            fling_r[:, 0] = lift_r[:, 0] * getattr(fling_forward_cfg, "x_scale", 1.)
            fling_r[:, 1] = fling_forward_cfg.y
            fling_r[:, 2] = fling_forward_cfg.h + self._table_height

            xyz_c = self._calculate_xyz_c(fling_l, fling_r)
            self._set_gripper_target_wrap(xyz=fling_l, hand_name="left", use_ik_init_cfg=True, xyz_c=xyz_c, leg_joint_cfg=fling_forward_cfg.leg)
            self._set_gripper_target_wrap(xyz=fling_r, hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_c, leg_joint_cfg=fling_forward_cfg.leg)
            self._move_to_robot_target_qpos(fling_forward_cfg.time, cache=True)
            if hasattr(fling_forward_cfg, "wait"):
                self._move_to_robot_target_qpos(fling_forward_cfg.wait, cache=True)
        self._api_run_all_cache()
        return fling_l, fling_r

    def _sequence_reset(self):
        self._api_manipulate_gripper("open", "open")
        self._api_move_to_qpos(self._init_qpos, cache=False)
    
    def _primitive_fling(
        self, 
        center_xy: torch.Tensor, 
        distance: torch.Tensor, 
        angle_degree: torch.Tensor,
        info: Dict[Literal["action_direct_ij_raw", "depth"], torch.Tensor],
    ):
        logger.info(f"primitive_fling raw:\n{center_xy}\n{distance}\n{angle_degree}")
        assert center_xy.shape == (1, 2)
        assert distance.shape == (1, )
        assert angle_degree.shape == (1, )
        fling_cfg = self._cfg.primitive.fling

        # clip input
        center_xy, distance, angle_degree = clip_fling_action(
            fling_cfg.action_space,
            center_xy, distance, angle_degree, self._dtype, self._device,
        )
        logger.info(f"primitive_fling:\n{center_xy}\n{distance}\n{angle_degree}")

        xyz_c = F.pad(center_xy, (0, 1), "constant", 0)
        xyz_c[:, 2] = self._table_height

        # center to left and right
        xyz_l = xyz_c.clone()
        xyz_l[:, 0] -= torch.cos(torch.deg2rad(angle_degree)) * distance / 2.
        xyz_l[:, 1] -= torch.sin(torch.deg2rad(angle_degree)) * distance / 2.

        xyz_r = xyz_c.clone()
        xyz_r[:, 0] += torch.cos(torch.deg2rad(angle_degree)) * distance / 2.
        xyz_r[:, 1] += torch.sin(torch.deg2rad(angle_degree)) * distance / 2.

        logger.info(f"primitive_fling decoded:\n{xyz_l}\n{xyz_r}")

        left_ij = FunnelActEncDec.decode_fling_direct_ij(info["action_direct_ij_raw"], "left")[0, :]
        right_ij = FunnelActEncDec.decode_fling_direct_ij(info["action_direct_ij_raw"], "right")[0, :]
        max_val = fling_cfg.max_pick_points_h_lower_offset
        self._sequence_move_to_pick_points(
            xyz_l, xyz_r, fling_cfg.pick_points,
            h_lower_l=np.clip(self._reproject_delta_depth - info["depth"][left_ij[0], left_ij[1]], 0., max_val),
            h_lower_r=np.clip(self._reproject_delta_depth - info["depth"][right_ij[0], right_ij[1]], 0., max_val),
        )
        fling_l, fling_r = self._sequence_liftup_flingforward(
            xyz_l, xyz_r,
            fling_cfg.lift_up,
            fling_cfg.fling_forwards,
        )
        self._api_manipulate_gripper("open", "open")
        self._sequence_move_to_h(fling_cfg.move_to_h, fling_l, fling_r)
        self._sequence_reset()
    
    def _primitive_find_and_pick_hanger(self, skip_query=False):
        logger.info("[STAGE] find and pick hanger")
        find_hanger_cfg = self._cfg.primitive.find_hanger
        while True:
            if find_hanger_cfg.hanger_pc_path is not None:
                logger.warn("overwrite hanger point cloud !")
                hanger_pc_np = trimesh.load_mesh(utils.get_path_handler()(find_hanger_cfg.hanger_pc_path)).vertices
            else:
                hanger_pc_np = self._get_hanger_point_cloud()
            hanger_pc = self._tensor(sample_point_cloud(hanger_pc_np, find_hanger_cfg.pc_num))
            
            mat = align_point_cloud(hanger_pc, self._hanger_surface, dtype=self._dtype, device=self._device)
            logger.info(f"hanger mat:\n{mat}")
            real_utils.vis.vis_pc(
                np.concatenate([
                    (utils.torch_to_numpy(self._hanger_surface) @ mat[:3, :3].T + mat[:3, 3]) * np.array([1., 1., 0.]), 
                    utils.torch_to_numpy(hanger_pc) * np.array([1., 1., 0.]), 
                ], axis=0), 
                np.concatenate([
                    np.array([[1., 0., 0.]] * self._hanger_surface.shape[0]), 
                    np.array([[0., 1., 0.]] * hanger_pc.shape[0])
                ], axis=0)
            )
            if skip_query:
                break
            break_flag = False
            while True:
                i = input("press c to continue, r to re-infer\n")
                if i == "c":
                    break_flag = True
                    break
                elif i == "r":
                    break_flag = False
                    break
            if break_flag:
                break

        xyz_r = self._tensor(mat[None, :3, 3])
        xyz_r[:, 2] = self._table_height
        rot = self._tensor((mat @ tra.euler_matrix(np.pi / 4, 0., 0.))[None, :3, :3], )

        logger.info("[STAGE] move to h_upper")
        self._set_gripper_target_wrap(
            xyz=xyz_r + self._tensor(
                [0., 0., find_hanger_cfg.h_upper]
            ), hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_r, leg_joint_cfg=find_hanger_cfg.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(find_hanger_cfg.time[0])

        logger.info("[STAGE] move to h_lower")
        self._set_gripper_target_wrap(
            xyz=xyz_r + self._tensor(
                [0., 0., find_hanger_cfg.h_lower]
            ), hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_r, leg_joint_cfg=find_hanger_cfg.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(find_hanger_cfg.time[1])
        
        logger.info("[STAGE] pick hanger")
        rpy_cfg = self._cfg.primitive.insert.press.hanger_rpy
        mat = np.linalg.inv(tra.euler_matrix(*(rpy_cfg.ee_tcp))) @ tra.euler_matrix(*(rpy_cfg.absolute))
        self._api_manipulate_gripper("open", "close")
        self._visualizer_set_hanger_mode("right", mat)

        logger.info("[STAGE] move to xyh_later")
        xyz_r = self._tensor((np.array(find_hanger_cfg.xyh_later) + np.array([0., 0., self._table_height]))[None, ...])
        rot = self._tensor(tra.euler_matrix(*(rpy_cfg.ee_tcp))[None, :3, :3])
        self._set_gripper_target_wrap(
            xyz=xyz_r, hand_name="right", use_ik_init_cfg=True, xyz_c=xyz_r, leg_joint_cfg=find_hanger_cfg.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(find_hanger_cfg.time[2])

        self._policy_e2e_cache["left_gripper"] = float(GRASP_NONE)
        self._policy_e2e_cache["right_gripper"] = float(GRASP_HANGER) # grasp hanger

    def _primitive_put_hanger_on_rack(self):
        hanger_on_rack = self._cfg.primitive.hanger_on_rack
        
        xyz_l = self._tensor(hanger_on_rack.xyh_l)[None, ...]
        xyz_l[:, 2] += self._table_height
        rot = self._tensor(tra.euler_matrix(*hanger_on_rack.rpy)[None, :3, :3])
        self._set_gripper_target_wrap(xyz=xyz_l, hand_name="left", use_ik_init_cfg=True, xyz_c=xyz_l, rot=rot)
        self._move_to_robot_target_qpos(hanger_on_rack.time)
        
        if FINAL_SHOW:
            qpos = {k: v.clone() for k, v in self._robot_target_qpos.items()}
            qpos["leg_joint4"] = hanger_on_rack.leg.leg_joint4.val
            self._api_move_to_qpos(qpos)
        
        while True:
            ans = input("enter 'finish' to open gripper and reset\n")
            if ans == "finish":
                break
            else:
                print(f"invalid input {ans}")
        
        self._api_manipulate_gripper("open", "open")
        self._visualizer_set_hanger_mode("none")
        self._sequence_reset()
    
    def _primitive_press(self, xy: torch.Tensor):
        logger.info(f"primitive_press raw:\n{xy}")
        assert xy.shape == (1, 2)
        press_cfg = self._cfg.primitive.insert.press
        
        xy = xy + self._tensor(press_cfg.offset)

        # clip input
        xyz_left_end = F.pad(torch.clip(
            xy.to(dtype=self._dtype).to(device=self._device),
            min=self._tensor([press_cfg.action_space.min]),
            max=self._tensor([press_cfg.action_space.max]),
        ), (0, 1), "constant", 0)
        xyz_left_end[:, 2] = self._table_height
        logger.info(f"primitive_press xyz:\n{xyz_left_end}")

        logger.info("[STAGE] move to xyz_left_end")
        xyz = self._endpoint_xyz_to_hanger_xyz(
            "left", xyz_left_end,
            np.array(press_cfg.hanger_rpy.absolute),
        )
        rot = self._tensor(tra.euler_matrix(*(press_cfg.hanger_rpy.ee_tcp))[None, :3, :3])
        for h_idx, h_str in enumerate(["h_upper", "h_inter", "h_lower"]):
            xyz_curr = xyz + self._tensor([0., 0., getattr(press_cfg.press, h_str)])
            self._set_gripper_target_wrap(
                xyz_curr, "right", 
                use_ik_init_cfg=bool(h_idx == 0), xyz_c=xyz, leg_joint_cfg=press_cfg.press.leg, rot=rot,
            )
            self._move_to_robot_target_qpos(press_cfg.press.time[h_idx])
        self._policy_cache["xyz_curr"] = xyz_curr.clone()
        self._policy_cache["rot"] = rot.clone()

    def _primitive_lift(self, xy: torch.Tensor):
        logger.info(f"primitive_lift raw:\n{xy}")
        assert xy.shape == (1, 2)
        lift_cfg = self._cfg.primitive.insert.lift

        # clip input
        xyz = F.pad(torch.clip(
            xy.to(dtype=self._dtype).to(device=self._device),
            min=self._tensor([lift_cfg.action_space.min]),
            max=self._tensor([lift_cfg.action_space.max]),
        ), (0, 1), "constant", 0)
        xyz[:, 2] = self._table_height
        logger.info(f"primitive_lift xyz:\n{xyz}")

        logger.info("[STAGE] move and lift")
        rot = self._tensor(tra.euler_matrix(*lift_cfg.pick_points_rpy)[None, :3, :3])
        self._sequence_move_to_pick_points(xyz, None, lift_cfg.pick_points, rot_l=rot)
        
        # rotate end effector
        # tf = self._urdf.link_transform_map[getattr(self._robot_cfg.grasp, "left").link]
        # self._set_gripper_target_wrap(
        #     xyz = tf[:, :3, 3] + self._tensor(lift_cfg.lift_rotate.xyz), 
        #     rot=self._tensor((
        #         utils.torch_to_numpy(tf[0, ...]) @ tra.euler_matrix(*lift_cfg.lift_rotate.rpy)
        #     )[None, :3, :3]), 
        #     hand_name="left", use_ik_init_cfg=False, 
        #     xyz_c=xyz, leg_joint_cfg=lift_cfg.pick_points.leg,
            
        # )
        # self._move_to_robot_target_qpos(lift_cfg.lift_rotate.time)

        # deprecate insert actions
        # the primitive_name and action_name use the old names

        press_cfg = self._cfg.primitive.insert.press
        xyz_curr: torch.Tensor = self._policy_cache.pop("xyz_curr") # [B, 3]
        rot: torch.Tensor = self._policy_cache.pop("rot") # [B, 3, 3]

        logger.info("[STAGE] insert")
        xyz_final = xyz_curr.clone()
        xyz_final[:, :2] += self._tensor(press_cfg.insert.xy)
        xyz_final[:, 2] = self._table_height + press_cfg.insert.h
        self._policy_cache["hanger_xy"] = utils.torch_to_numpy(xyz_final)[0, 0:2]
        with open(HANGER_XY_CACHE_FILE, "w") as f_obj:
            json.dump(dict(hanger_xy=self._policy_cache["hanger_xy"].tolist()), f_obj, indent=4)

        # insert step 1
        final_coeff = 0.5
        xyz_r = xyz_curr * (1.0 - final_coeff) + xyz_final * final_coeff
        self._set_gripper_target_wrap(
            xyz_r, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=press_cfg.insert.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(press_cfg.insert.time[0])

        # move away left hand
        self._api_manipulate_gripper("open", "close")
        xyz_l = self._get_end_effector_mat("left")[:, :3, 3] + self._tensor(press_cfg.insert.xyz_delta_left)   
        self._set_gripper_target_wrap(
            xyz_l, "left", 
            use_ik_init_cfg=False, xyz_c=xyz_l, leg_joint_cfg=press_cfg.insert.leg, 
        )
        self._move_to_robot_target_qpos(press_cfg.insert.time[1])

        # insert step 2
        xyz_r = xyz_final
        self._set_gripper_target_wrap(
            xyz_r, "right", 
            use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=press_cfg.insert.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(press_cfg.insert.time[2])
        
        self._api_manipulate_gripper("open", "open")
        self._visualizer_set_hanger_mode("none")
        self._sequence_move_to_h(press_cfg.move_to_h, self._get_end_effector_mat("left")[:, :3, 3], self._get_end_effector_mat("right")[:, :3, 3])
        self._sequence_reset()
    
    def _primitive_drag(self, xy: torch.Tensor):
        logger.info(f"primitive_drag raw:\n{xy}")
        assert xy.shape == (1, 2)
        drag_cfg = self._cfg.primitive.insert.drag
        
        xy = xy + self._tensor(drag_cfg.offset)

        # clip input
        xyz = F.pad(torch.clip(
            xy.to(dtype=self._dtype).to(device=self._device),
            min=self._tensor([drag_cfg.action_space.min]),
            max=self._tensor([drag_cfg.action_space.max]),
        ), (0, 1), "constant", 0)
        xyz[:, 2] = self._table_height
        logger.info(f"primitive_drag xyz:\n{xyz}")

        logger.info("[STAGE] regrasp")
        xyz_l = self._tensor([[*self._policy_cache.pop("hanger_xy"), self._table_height]])
        xyz_l[:, :2] += self._tensor(drag_cfg.regrasp.offset)
        rot = self._tensor(tra.euler_matrix(*(drag_cfg.hanger_rpy.ee_tcp))[None, :3, :3])
        hanger_origin = np.linalg.inv(tra.euler_matrix(*(drag_cfg.hanger_rpy.ee_tcp))) @ tra.euler_matrix(*(drag_cfg.hanger_rpy.absolute))
        # add one more step to avoid collision between arm and hanger
        self._set_gripper_target_wrap(
            xyz_l + self._tensor(
                [0., 0., drag_cfg.regrasp.h_upper]
            ), hand_name="left", use_ik_init_cfg=True, xyz_c=xyz_l, 
            leg_joint_cfg=drag_cfg.regrasp.leg, rot=rot,
        )
        self._move_to_robot_target_qpos(drag_cfg.regrasp.time[0])
        
        self._set_gripper_target_wrap(
            xyz_l + self._tensor(
                [0., 0., drag_cfg.regrasp.h_inter]
            ), hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_l, 
            leg_joint_cfg=drag_cfg.regrasp.leg, rot=rot,
        )
        self._move_to_robot_target_qpos(drag_cfg.regrasp.time[1])
        
        self._set_gripper_target_wrap(
            xyz_l + self._tensor(
                [0., 0., drag_cfg.regrasp.h_lower]
            ), hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_l, 
            leg_joint_cfg=drag_cfg.regrasp.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(drag_cfg.regrasp.time[2])
        self._api_manipulate_gripper("close", "open")
        self._visualizer_set_hanger_mode("left", hanger_origin)

        logger.info("[STAGE] slightly lift the hanger")
        self._set_gripper_target_wrap(
            xyz_l + self._tensor(
                [0., 0., drag_cfg.regrasp.h_hanger_upper]
            ), "left", use_ik_init_cfg=False, xyz_c=xyz_l, leg_joint_cfg=drag_cfg.regrasp.leg, rot=rot, 
        )
        self._move_to_robot_target_qpos(drag_cfg.regrasp.time[3])

        logger.info("[STAGE] pick")
        self._sequence_move_to_pick_points(None, xyz, drag_cfg.pick_points)

        logger.info("[STAGE] drag")
        # xyz_r = xyz + self._tensor([0., 0., drag_cfg.pick_points.h_later])
        hanger_mat = self._tensor([tra.translation_matrix(
            utils.torch_to_numpy(xyz_l + self._tensor([0., 0., drag_cfg.regrasp.h_hanger_upper]))[0, ...]
        ) @ tra.euler_matrix(*(drag_cfg.hanger_rpy.absolute))])
        for drag_step_cfg in drag_cfg.drag:
            xyz_r = self._get_current_xyz("right", hanger_mat)
            xyz_r[:, 2] = self._table_height
            xyz_r[:, :] += self._tensor(drag_step_cfg.right.xyh)
            self._set_gripper_target_wrap(
                xyz_r, "right", 
                use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=drag_step_cfg.leg
            )
            if hasattr(drag_step_cfg, "left"):
                self._set_gripper_target_wrap(
                    xyz_l + self._tensor(
                        drag_step_cfg.left.xyh
                    ), "left", use_ik_init_cfg=False, xyz_c=xyz_l, 
                    leg_joint_cfg=drag_step_cfg.leg, 
                    rot=self._tensor(tra.euler_matrix(*(drag_step_cfg.left.rpy))[None, :3, :3]) @ rot, 
                )
            self._move_to_robot_target_qpos(drag_step_cfg.time)
    
    def _primitive_rotate(self, xy: torch.Tensor):
        logger.info(f"primitive_rotate raw:\n{xy}")
        assert xy.shape == (1, 2)
        rotate_cfg = self._cfg.primitive.insert.rotate

        # clip input
        xyz_l = F.pad(torch.clip(
            xy.to(dtype=self._dtype).to(device=self._device),
            min=self._tensor([rotate_cfg.action_space.min]),
            max=self._tensor([rotate_cfg.action_space.max]),
        ), (0, 1), "constant", 0)
        xyz_l[:, 2] = self._table_height 
        logger.info(f"primitive_rotate xyz:\n{xyz_l}")

        logger.info("[STAGE] rotate")
        rot = self._tensor(tra.euler_matrix(*(rotate_cfg.hanger_rpy.ee_tcp))[None, :3, :3])
        xyz_l_final = self._endpoint_xyz_to_hanger_xyz(
            "right", xyz_l, np.array(rotate_cfg.hanger_rpy.absolute),
        ) + self._tensor([0., 0., float(rotate_cfg.rotate.h)])
        self._set_gripper_target_wrap(
            xyz_l_final, "left", use_ik_init_cfg=False, 
            xyz_c=xyz_l_final, leg_joint_cfg=rotate_cfg.rotate.leg, rot=rot,
        )
        self._move_to_robot_target_qpos(rotate_cfg.rotate.time, cache=True)
        self._api_run_all_cache()

        xyz_h_final = xyz_l_final.clone()
        xyz_h_final[:, 0] = rotate_cfg.pull.hanger.x
        xyz_h_final[:, 1] = rotate_cfg.pull.hanger.y
        self._set_gripper_target_wrap(
            xyz_h_final, "left", use_ik_init_cfg=False, 
            xyz_c=xyz_h_final, leg_joint_cfg=rotate_cfg.pull.leg, rot=rot
        )
        self._move_to_robot_target_qpos(rotate_cfg.pull.time)
        
        logger.info("[STAGE] real pull")
        self._api_manipulate_gripper("close", "open")
        xyz_r = self._get_end_effector_mat("right")[:, :3, 3]
        xyz_r[:, 2] = self._table_height + rotate_cfg.pull.h_upper
        self._set_gripper_target_wrap(xyz_r, "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        
        xyz_r = xyz_h_final.clone()
        xyz_r[:, 2] = self._table_height + rotate_cfg.pull.h_upper
        xyz_r[:, 1] += rotate_cfg.pull.start.y
        xyz_r[:, 0] += rotate_cfg.pull.start.x
        self._set_gripper_target_wrap(xyz_r + self._tensor([rotate_cfg.pull.mid_point]), "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        
        self._set_gripper_target_wrap(xyz_r, "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        
        xyz_r[:, 2] = self._table_height + rotate_cfg.pull.h_lower
        self._set_gripper_target_wrap(xyz_r, "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        self._api_manipulate_gripper("close", "close")
        xyz_r[:, 2] = self._table_height + rotate_cfg.pull.h_upper
        self._set_gripper_target_wrap(xyz_r, "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        
        xyz_r = xyz_h_final.clone()
        xyz_r[:, 2] = self._table_height + rotate_cfg.pull.h_upper
        xyz_r[:, 1] += rotate_cfg.pull.end.y
        xyz_r[:, 0] += rotate_cfg.pull.end.x
        self._set_gripper_target_wrap(xyz_r, "right", use_ik_init_cfg=False, xyz_c=xyz_r, leg_joint_cfg=rotate_cfg.pull.leg)
        self._move_to_robot_target_qpos(time=rotate_cfg.pull.time)
        
        self._api_manipulate_gripper("close", "open")

    def _primitive_keypoints(
        self, 
        xy_clothes_l: torch.Tensor,
        xy_clothes_r: torch.Tensor,
    ):
        logger.info(f"primitive_keypoints raw:\n{xy_clothes_l}\n{xy_clothes_r}")
        assert xy_clothes_l.shape == (1, 2)
        assert xy_clothes_r.shape == (1, 2)

        keypoint_cfg = self._cfg.primitive.keypoint
        xy_clothes_l, xy_clothes_r = self._keypoints_clip_input(
            xy1=xy_clothes_l, xy2=xy_clothes_r,
            **(omegaconf.OmegaConf.to_container(keypoint_cfg.action_space))
        )
        logger.info(f"primitive_keypoints:\n{xy_clothes_l}\n{xy_clothes_r}")

        xyz_clothes_l = F.pad(xy_clothes_l, (0, 1), "constant", 0)
        xyz_clothes_l[:, 2] = self._table_height

        xyz_clothes_r = F.pad(xy_clothes_r, (0, 1), "constant", 0)
        xyz_clothes_r[:, 2] = self._table_height

        xy_clothes_c = (xy_clothes_l + xy_clothes_r) / 2.
        l_to_r = xy_clothes_r - xy_clothes_l
        theta1 = float(torch.atan2(xy_clothes_c[:, 1], xy_clothes_c[:, 0]))
        theta2 = float(torch.atan2(l_to_r[:, 1], l_to_r[:, 0]))
        theta3 = theta2  - theta1
        while theta3 < 0:
            theta3 += 2 * math.pi
        exchange_hand = (theta3 > math.radians(40) and theta3 < math.radians(140))
        # exchange_hand = bool(xyz_clothes_l[:, 0] > xyz_clothes_r[:, 0])
        if not exchange_hand:
            xyz_l = xyz_clothes_l.clone()
            xyz_r = xyz_clothes_r.clone()
        else:
            exchange_cfg = keypoint_cfg.exchange_hand
            self._sequence_move_to_pick_points(xyz_clothes_r, xyz_clothes_l, exchange_cfg.pick_points)
            d = (xyz_clothes_r - xyz_clothes_l).norm(dim=1)

            xyz_l_mid = self._tensor([[0., 0.5, self._table_height]])
            xyz_r_mid = self._tensor([[0., 0.5, self._table_height]])
            
            '''if bool(xyz_clothes_l[:, 1] > xyz_clothes_r[:, 1]):
                xyz_l_mid[:, 1] -= d / 2
                xyz_r_mid[:, 1] += d / 2
                x_upper = -exchange_cfg.move_to_target.x_upper_abs
            else:
                xyz_r_mid[:, 1] -= d / 2
                xyz_l_mid[:, 1] += d / 2
                x_upper = +exchange_cfg.move_to_target.x_upper_abs'''

            xyz_r_mid[:, 1] -= d / 2
            xyz_l_mid[:, 1] += d / 2
            x_upper = +exchange_cfg.move_to_target.x_upper_abs
            logger.info(f"primitive_keypoints exchange:{xyz_r_mid} {xyz_l_mid}")

            h_upper = exchange_cfg.move_to_target.h_upper
            h_put = exchange_cfg.move_to_target.h_put
            
            offset = self._tensor([0., 0., h_upper])
            xyz_c_mid = (xyz_l_mid + xyz_r_mid) / 2 + offset
            self._set_gripper_target_wrap(xyz=xyz_l_mid + offset, hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._set_gripper_target_wrap(xyz=xyz_r_mid + offset, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._move_to_robot_target_qpos(exchange_cfg.move_to_target.time[0], cache=False)

            offset = self._tensor([x_upper, 0., h_upper])
            xyz_c_mid = (xyz_l_mid + xyz_r_mid) / 2 + offset
            self._set_gripper_target_wrap(xyz=xyz_l_mid + offset, hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._set_gripper_target_wrap(xyz=xyz_r_mid + offset, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._move_to_robot_target_qpos(exchange_cfg.move_to_target.time[1], cache=True)

            offset = self._tensor([0., 0., h_put])
            xyz_c_mid = (xyz_l_mid + xyz_r_mid) / 2 + offset
            self._set_gripper_target_wrap(xyz=xyz_l_mid + offset, hand_name="left", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._set_gripper_target_wrap(xyz=xyz_r_mid + offset, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz_c_mid)
            self._move_to_robot_target_qpos(exchange_cfg.move_to_target.time[2], cache=True)

            self._api_run_all_cache()

            self._api_manipulate_gripper("open", "open")
            self._sequence_move_to_h(exchange_cfg.move_to_h, xyz_l_mid, xyz_r_mid)
            self._sequence_reset()

            xyz_l = xyz_r_mid.clone()
            xyz_r = xyz_l_mid.clone()

        self._sequence_move_to_pick_points(xyz_l, xyz_r, keypoint_cfg.fling.pick_points)
        fling_l, fling_r = self._sequence_liftup_flingforward(
            xyz_l, xyz_r,
            keypoint_cfg.fling.lift_up,
            keypoint_cfg.fling.fling_forwards,
        )
        self._api_manipulate_gripper("open", "open")
        self._sequence_move_to_h(keypoint_cfg.fling.move_to_h, fling_l, fling_r)
        self._sequence_reset()
    
    def _primitive_hanger_right_to_left(self, xy: torch.Tensor):
        leg_cfg = self._cfg.primitive.insert.drag.regrasp.leg
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.10)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz, leg_joint_cfg=None, rot=self._tensor(tra.euler_matrix(np.pi / 4, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.01)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz, leg_joint_cfg=leg_cfg, rot=self._tensor(tra.euler_matrix(np.pi / 4, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        self._api_manipulate_gripper("open", "open")
        self._visualizer_set_hanger_mode("none")
        
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.10)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="right", use_ik_init_cfg=False, xyz_c=xyz, leg_joint_cfg=leg_cfg, rot=self._tensor(tra.euler_matrix(np.pi / 4, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        self._sequence_reset()
        
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.20)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="left", use_ik_init_cfg=True, xyz_c=xyz, leg_joint_cfg=leg_cfg, rot=self._tensor(tra.euler_matrix(0, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.01)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="left", use_ik_init_cfg=False, xyz_c=xyz, leg_joint_cfg=leg_cfg, rot=self._tensor(tra.euler_matrix(0, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        self._api_manipulate_gripper("close", "open")
        self._visualizer_set_hanger_mode("left", np.eye(4))
        
        xyz = F.pad(xy, (0, 1), "constant", self._table_height + 0.10)
        self._set_gripper_target_wrap(xyz=xyz, hand_name="left", use_ik_init_cfg=False, xyz_c=xyz, leg_joint_cfg=leg_cfg, rot=self._tensor(tra.euler_matrix(0, 0, np.pi / 4)[None, :3, :3]))
        self._move_to_robot_target_qpos()
        
        self._sequence_reset()
    
    def _clamp_min_along_direction(self, xy: torch.Tensor, angle_degree: float, min_val: float):
        """
        xy: [B, 2]
        """
        B, _ = xy.shape
        n = self._tensor([math.cos(math.radians(angle_degree)), math.sin(math.radians(angle_degree))])
        xy_1 = xy - (xy * n).sum(dim=1, keepdim=True) * n
        xy_2 = (xy * n).sum(dim=1, keepdim=True).clamp_min(min_val) * n
        return xy_1 + xy_2

    def _precheck_lift_press(self, xyl: torch.Tensor, xyp: torch.Tensor):
        """Only precheck the positional relationship between `xy1` and `xy2`, do not clip according to the action space."""
        logger.info(f"\n\n\nprecheck_lift_press {xyl} {xyp}")
        mid = (xyl + xyp) / 2
        dif = xyl - xyp
        dif_dir = dif / dif.norm(dim=1, keepdim=True).clamp_min(1e-6)
        dif = dif_dir * dif.norm(dim=1, keepdim=True).clamp_min(0.10)
        xyl = mid + dif / 2
        xyp = mid - dif / 2
        logger.info(f"precheck_lift_press modified: {xyl} {xyp}\n\n")
        return xyl, xyp
        mid = (xyl + xyp) / 2
        dif = xyl - xyp
        dif = self._clamp_min_along_direction(dif, 225, 0.04)        
        xyp, xyl = mid - dif / 2, mid + dif / 2
        logger.info(f"precheck_lift_press modified: {xyl} {xyp}\n\n")
        return xyl, xyp

    def _precheck_drag_rotate(self, xyd: torch.Tensor, xyr: torch.Tensor):
        """Only precheck the positional relationship between `xy1` and `xy2`, do not clip according to the action space."""
        logger.info(f"\n\n\nprecheck_drag_rotate {xyd} {xyr}")
        logger.info(f"precheck_drag_rotate modified: {xyd} {xyr}\n\n")
        return xyd, xyr
        base_point = self._tensor([self._hanger_xy])
        dif = xyd - base_point
        dif = self._clamp_min_along_direction(dif, 315, 0.12)
        xyd = base_point + dif

    def _inference_fling(self, depth_raw, mask_garment):
        log_path = os.path.abspath(f"infer/fling/{self._inference_cnt['f']}")
        self._inference_cnt['f'] += 1
        ret = self._policy.predict_fling(depth_raw, mask_garment, dict(log_path=log_path))
        real_utils.vis.show(np.asarray(convert_from_path(os.path.join(log_path, "0.pdf"))[0]))
        return ret
    
    def _inference_insert_left(self, depth_raw, mask_garment, mask_hanger, mask_neckline, rgb=None):
        log_path = os.path.abspath(f"infer/insert_left/{self._inference_cnt['il']}")
        self._inference_cnt['il'] += 1
        ret = self._policy.predict_insert(
            depth_raw, mask_garment, mask_hanger, mask_neckline, 
            self._get_real_reproject_camera_z(), "left", dict(log_path=log_path)
        )
        os.makedirs(log_path, exist_ok=True)
        if os.path.exists(os.path.join(log_path, "0.pdf")):
            real_utils.vis.show(np.asarray(convert_from_path(os.path.join(log_path, "0.pdf"))[0]))
        if rgb is not None:
            Image.fromarray(rgb).save(os.path.join(log_path, "rgb.png"))
        return ret
        
    def _inference_insert_right(self, depth_raw, mask_garment, mask_hanger, mask_neckline, rgb=None):
        log_path = os.path.abspath(f"infer/insert_right/{self._inference_cnt['ir']}")
        self._inference_cnt['ir'] += 1
        ret = self._policy.predict_insert(
            depth_raw, mask_garment, mask_hanger, mask_neckline, 
            self._get_real_reproject_camera_z(), "right", dict(log_path=log_path)
        )
        os.makedirs(log_path, exist_ok=True)
        if os.path.exists(os.path.join(log_path, "0.pdf")):
            real_utils.vis.show(np.asarray(convert_from_path(os.path.join(log_path, "0.pdf"))[0]))
        if rgb is not None:
            Image.fromarray(rgb).save(os.path.join(log_path, "rgb.png"))
        return ret
    
    def _inference_keypoints(self, depth_raw, mask_garment, rgb=None):
        log_path = os.path.abspath(f"infer/keypoints/{self._inference_cnt['kp']}")
        self._inference_cnt['kp'] += 1
        ret = self._policy.predict_keypoints(
            depth_raw, mask_garment, 
            self._get_real_reproject_camera_z(), dict(log_path=log_path)
        )
        vis_img = np.asarray(convert_from_path(os.path.join(log_path, "0.pdf"))[0])
        real_utils.vis.show(vis_img)
        Image.fromarray(vis_img).save(os.path.join(log_path, "0.png"))
        if rgb is not None:
            Image.fromarray(rgb).save(os.path.join(log_path, "rgb.png"))
        return ret
    
    def _e2e_get_obs_and_sta(self) -> Dict[Literal["depth_raw", "mask_garment", "mask_hanger", "robot_state"], np.ndarray]:
        obs = self._real_api.get_obs()
        # fix depth
        e2e_fix_depth(
            obs["depth"], self._get_head_camera_extrinsics(), self._get_head_camera_intrinsics(),
            self._cfg.obs.reproject.input_max_y, 
            self._cfg.obs.reproject.input_min_y, 
            self._cfg.obs.reproject.input_max_x_abs, 
            self._table_height, MAX_DEPTH,
        )
        
        # visualize
        title_list = ['depth', 'clothes', 'hanger']
        fig = plot_wrap_fig(
            [obs["rgb"][None, ...]] + [obs[s][None, ...] for s in title_list], 
            ["rgb"] + title_list, [None] + ["gray"] * 3, 
            plot_batch_size=1, width_unit=6.
        )
        fig.canvas.draw()
        vis_img = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='').reshape(fig.canvas.get_width_height()[::-1] + (3,))
        real_utils.vis.show(vis_img)

        robot_state = np.zeros(2 + 7 + 7 + 4)
        robot_state[0] = self._policy_e2e_cache["left_gripper"]
        robot_state[1] = self._policy_e2e_cache["right_gripper"]
        for joint_idx, joint_name in enumerate(self._cfg.obj.robot.e2e.joints):
            robot_state[joint_idx + 2] = float(self._urdf.cfg[joint_name])

            if hasattr(self._cfg.obj.robot.e2e.joints_net_input_offset, joint_name):
                robot_state[joint_idx + 2] += getattr(self._cfg.obj.robot.e2e.joints_net_input_offset, joint_name)
            if hasattr(self._cfg.obj.robot.e2e.joints_net_input_overwrite, joint_name):
                robot_state[joint_idx + 2] = getattr(self._cfg.obj.robot.e2e.joints_net_input_overwrite, joint_name)
        
        return dict(
            depth_raw=obs["depth"],
            mask_garment=obs["clothes"],
            mask_hanger=obs["hanger"],
            robot_state=robot_state,
        )
        
    def _e2e_predict(self, **obs_kwargs):
        action = self._policy.get_action_e2e(**obs_kwargs)
        return action

    def _e2e_move_gripper_to_pick_garment(self, hand_name: str):
        logger.info(f"magic e2e_move_gripper_to_pick_garment {hand_name}")
        curr_qpos = {k: v.clone() for k, v in self._urdf.cfg.items()}
        
        mat = self._urdf.link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link]
        if torch.all(mat[:, 2, 3] > self._table_height + 0.15):
            logger.info(f"pick garment position too high ... {mat[:, 2, 3]}")
            return
        
        mat[:, 2, 3] = self._table_height + 0.01
        def err_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link].view(1, 16, 16) # [B, 16, 16]
            err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(1, 16) # [B, 16]
            return err_mat
        def loss_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[getattr(self._robot_cfg.grasp, hand_name).link]
            err_mat = curr_mat4 - mat # [B, 16]
            return torch.sum(torch.square(err_mat).view(1, 16), dim=1) # [B, ]
        new_cfg, info = self._urdf.inverse_kinematics_optimize(
            err_func=err_func,
            loss_func=loss_func,
            **(self._robot_ik_solver_cfg),
        )
        logger.info(pprint.pformat(curr_qpos))
        logger.info(pprint.pformat(new_cfg))
        
        self._api_move_to_qpos(qpos=new_cfg, overwrite_single_step=False, qpos_to_exec=self._fix_qpos.add_ee_offset_to_qpos(new_cfg))
        self._api_manipulate_gripper(
            left=("close" if hand_name == "left" else "none"), 
            right=("close" if hand_name == "right" else "none"),
            overwrite_single_step=False
        )
        # self._api_move_to_qpos(qpos=curr_qpos, overwrite_single_step=False, qpos_to_exec=self._fix_qpos.add_ee_offset_to_qpos(curr_qpos))
    
    def _e2e_release_hanger_step(self):
        logger.info(f"magic e2e_release_hanger_step")
        self._policy_e2e_cache["hanger_release"] = \
            self._urdf.link_transform_map[self._robot_cfg.grasp.right.link][0, :3, 3].clone()
    
    def _e2e_regrasp_hanger_step(self):
        logger.info(f"magic e2e_regrasp_hanger_step")
        if not "hanger_release" in self._policy_e2e_cache.keys():
            logger.info("e2e_regrasp_hanger_step hanger_release is None")
            return
        
        curr_qpos = {k: v.clone() for k, v in self._urdf.cfg.items()}
        mat = self._urdf.link_transform_map[self._robot_cfg.grasp.left.link]
        if torch.all(mat[:, 2, 3] > self._table_height + 0.15):
            logger.info(f"e2e_regrasp_hanger_step too high ... {mat[:, 2, 3]}")
            return
        mat[:, :3, 3] = self._policy_e2e_cache["hanger_release"]
        mat[:, 2, 3] = self._table_height + 0.01
        def err_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[self._robot_cfg.grasp.left.link].view(1, 16, 16) # [B, 16, 16]
            err_mat = curr_mat4[:, torch.arange(16), torch.arange(16)] - mat.view(1, 16) # [B, 16]
            return err_mat
        def loss_func(link_transform_map: Dict[str, torch.Tensor]):
            curr_mat4 = link_transform_map[self._robot_cfg.grasp.left.link]
            err_mat = curr_mat4 - mat # [B, 16]
            return torch.sum(torch.square(err_mat).view(1, 16), dim=1) # [B, ]
        new_cfg, info = self._urdf.inverse_kinematics_optimize(
            err_func=err_func,
            loss_func=loss_func,
            **(self._robot_ik_solver_cfg),
        )
        logger.info(pprint.pformat(curr_qpos))
        logger.info(pprint.pformat(new_cfg))
        
        self._api_move_to_qpos(qpos=new_cfg, overwrite_single_step=False, qpos_to_exec=self._fix_qpos.add_ee_offset_to_qpos(new_cfg))
        self._api_manipulate_gripper(left="close", right="none", overwrite_single_step=False)
        # self._api_move_to_qpos(qpos=curr_qpos, overwrite_single_step=False, qpos_to_exec=self._fix_qpos.add_ee_offset_to_qpos(curr_qpos))

    def _e2e_execute(self, action: np.ndarray):
        left_gripper_action = "open" if np.allclose(action[0], GRASP_NONE) else "close"
        right_gripper_action = "open" if np.allclose(action[1], GRASP_NONE) else "close"
        if np.allclose(action[0], GRASP_CLOTHES) and np.allclose(self._policy_e2e_cache["left_gripper"], GRASP_NONE):
            self._e2e_move_gripper_to_pick_garment("left")
        if np.allclose(action[1], GRASP_CLOTHES) and np.allclose(self._policy_e2e_cache["right_gripper"], GRASP_NONE):
            self._e2e_move_gripper_to_pick_garment("right")
        if np.allclose(action[0], GRASP_HANGER) and np.allclose(self._policy_e2e_cache["left_gripper"], GRASP_NONE):
            self._e2e_regrasp_hanger_step()
        if np.allclose(action[1], GRASP_NONE) and np.allclose(self._policy_e2e_cache["right_gripper"], GRASP_HANGER):
            self._e2e_release_hanger_step()
        self._api_manipulate_gripper(left=left_gripper_action, right=right_gripper_action, overwrite_single_step=False)
        self._policy_e2e_cache["left_gripper"] = float(action[0])
        self._policy_e2e_cache["right_gripper"] = float(action[1])

        qpos = dict()
        for joint_idx, joint_name in enumerate(self._cfg.obj.robot.e2e.joints):
            if hasattr(self._cfg.obj.robot.e2e.joints_net_input_overwrite, joint_name):
                dq = 0.
            else:
                dq = action[joint_idx + 2]
            qpos[joint_name] = float(self._urdf.cfg[joint_name]) + dq
        
        # fix qpos
        qpos = fix_joint_cfg(
            self._urdf, qpos, self._tensor([self._table_height]) + self._policy._cfg.insert.min_h,
            self._dtype, self._device, 1,
        )
        
        qpos_to_exec = self._fix_qpos.add_ee_offset_to_qpos(qpos) # fix qpos to execute
        qpos = self._to_full_cfg(qpos)
        self._api_move_to_qpos(qpos, overwrite_single_step=False, qpos_to_exec=qpos_to_exec)
    
    def _e2e_reset(self):
        self._policy.reset_e2e()
        self._policy_e2e_cache["left_gripper"] = float(GRASP_NONE)
        self._policy_e2e_cache["right_gripper"] = float(GRASP_NONE)
        self._policy_e2e_cache.pop("hanger_release", None)

    def run_human_control(self):
        self._sequence_reset()
        prompt = (
            "What to do next?\n"
            "r: reset\n"
            "q: quit\n"
            "p: pick points, p / none / 0.1, 0.5\n"
            "h: exchange hanger, h / -0.04, 0.56\n"
            "f: fling, f / 0.0, 0.5, 0.25, 30.\n"
            "il: insert left,  il / 0.00, 0.50, -0.05, 0.40\n"
            "ir: insert right, ir / +0.10, 0.40, 0.00, 0.50\n"
            "kp: keypoints, kp / -0.20, 0.45, -0.20, 0.55\n"
            "_f: fling by policy\n"
            "_il: insert left by policy\n"
            "_ir: insert right by policy\n"
            "_kp: keypoints by policy\n"
            "_ki: keypoints by policy with left and right hand exchanged\n"
            "_oi: only inference keypoints score\n"
            "_obs: get observation, '_obs best' or '_obs 0.475'\n"
            "_hanger: grasp hanger\n"
            "_e2e: run end2end policy\n"
            "_reset_e2e: reset end2end policy\n"
            "_infer_all: _kp + _il + _ir\n"
        )
        
        def parse_action(i: str, nargs: int):
            actions = i.split("/")
            if len(actions) != 2:
                return
            def parse_single_action(a: str):
                a = a.strip().lower().replace(" ", "")
                try:
                    x = [float(_) for _ in a.split(",")[:nargs]]
                    return self._tensor([x])
                except Exception as e:
                    print(e)
            return parse_single_action(actions[1])

        def iter_func():
            # obs = self._get_processed_obs(plot_fig=False)
            # print(f"\nkeypoints score: {self._policy.predict_keypoints_score(obs['depth'], obs['clothes'])}")
            i = input(prompt)
            if i == "r":
                self._sequence_reset()
            elif i == "q":
                real_utils.vis.stop_all()
                return 1
            elif i.startswith("p"):
                def parse_action_p(i: str):
                    actions = i.split("/")
                    if len(actions) != 3:
                        return
                    def parse_single_action(a: str):
                        a = a.strip().lower().replace(" ", "")
                        if a == "none":
                            pass
                        else:
                            try:
                                x, y = [float(_) for _ in a.split(",")[:2]]
                                return self._tensor([[x, y, self._table_height]])
                            except Exception as e:
                                print(e)
                    return parse_single_action(actions[1]), parse_single_action(actions[2])
                result = parse_action_p(i)
                if result is None:
                    print(f"invalid action {i}")
                else:
                    l, r = result
                    pick_points_cfg = omegaconf.DictConfig(dict(
                        h_upper=0.10,
                        h_lower=0.02,
                        h_later=0.10,
                        leg=dict(
                            leg_joint3=dict(val=0.5),
                            leg_joint4=dict(coeff=-1.8),
                        ),
                        time=[2., None, None],
                    ))
                    self._sequence_move_to_pick_points(l, r, pick_points_cfg)
            elif i.startswith("h"):
                result = parse_action(i, 2)
                if result is None:
                    print(f"invalid action {i}")
                else:
                    xy = result
                    self._primitive_find_and_pick_hanger()
                    self._primitive_hanger_right_to_left(xy)
            elif i.startswith("f"):
                result = parse_action(i, 4)
                if result is None:
                    print(f"invalid action {i}")
                else:
                    center_xy, distance, angle_degree = result[:, :2], result[:, 2], result[:, 3]
                    self._primitive_fling(center_xy, distance, angle_degree)
            elif i.startswith("il") or i.startswith("ir") or i.startswith("kp"):
                result = parse_action(i, 4)
                if result is None:
                    print(f"invalid action {i}")
                else:
                    xy1, xy2 = result[:, :2], result[:, 2:]
                    if i.startswith("il"):
                        self._primitive_find_and_pick_hanger()
                        xy1, xy2 = self._precheck_lift_press(xy1, xy2)
                        self._primitive_press(xy1)
                        self._primitive_lift(xy2)
                    elif i.startswith("ir"):
                        xy1, xy2 = self._precheck_drag_rotate(xy1, xy2)
                        self._primitive_drag(xy1)
                        self._primitive_rotate(xy2)
                        self._primitive_put_hanger_on_rack()
                    elif i.startswith("kp"):
                        self._primitive_keypoints(xy1, xy2)
                    else:
                        raise ValueError(i)
            elif i.startswith("_f"):
                while True:
                    obs = self._get_processed_obs()
                    infer = self._inference_fling(obs["depth"], obs["clothes"])
                    q = input("press c to continue, otherwise to re-infer\n")
                    if q == "c":
                        break
                self._primitive_fling(
                    infer["action"]["center_xy"], 
                    infer["action"]["distance"], 
                    infer["action"]["angle_degree"], 
                    dict(
                        action_direct_ij_raw=infer["info"]["action_direct_ij_raw"],
                        depth=obs["depth"],
                    )
                )
                obs = self._get_processed_obs()
                print(f"\nkeypoints score: {self._policy.predict_keypoints_score(obs['depth'], obs['clothes'])}")
            elif i.startswith("_il"):
                self._primitive_find_and_pick_hanger()
                xy1_offset, xy2_offset = self._tensor([0., 0.]), self._tensor([0., 0.])
                while True:
                    obs = self._get_processed_obs(seg_neckline=True)
                    infer = self._inference_insert_left(obs["depth"], obs["clothes"], obs["hanger"], obs["neckline"], obs["raw_rgb"])
                    xy1, xy2 = self._precheck_lift_press(infer["action"]["action_1_xy"], infer["action"]["action_2_xy"])
                    break_flag = False
                    while True:
                        q = input("press c to continue, r to reinfer, o / x1, y1, x2, y2 to add offset, otherwise to re-infer\n")
                        if q == "c":
                            break_flag = True
                            break
                        elif q.startswith("o"):
                            result = parse_action(q, 4)
                            if result is None:
                                print(f"invalid action {q}")
                            else:
                                xy1_offset[...] = result[:, :2]
                                xy2_offset[...] = result[:, 2:]
                                break_flag = True
                                break
                        elif q == "r":
                            break_flag = False
                            break
                    if break_flag: break
                self._primitive_press(xy1 + xy1_offset)
                self._primitive_lift(xy2 + xy2_offset)
            elif i.startswith("_ir"):
                xy1_offset, xy2_offset = self._tensor([0., 0.]), self._tensor([0., 0.])
                while True:
                    obs = self._get_processed_obs()
                    infer = self._inference_insert_right(obs["depth"], obs["clothes"], obs["hanger"], obs["neckline"], obs["raw_rgb"])
                    xy1, xy2 = self._precheck_drag_rotate(infer["action"]["action_1_xy"], infer["action"]["action_2_xy"])
                    break_flag = False
                    while True:
                        q = input("press c to continue, r to reinfer, o / x1, y1, x2, y2 to add offset, otherwise to re-infer\n")
                        if q == "c":
                            break_flag = True
                            break
                        elif q.startswith("o"):
                            result = parse_action(q, 4)
                            if result is None:
                                print(f"invalid action {q}")
                            else:
                                xy1_offset[...] = result[:, :2]
                                xy2_offset[...] = result[:, 2:]
                                break_flag = True
                                break
                        elif q == "r":
                            break_flag = False
                            break
                    if break_flag: break
                self._primitive_drag(xy1 + xy1_offset)
                self._primitive_rotate(xy2 + xy2_offset)
                self._primitive_put_hanger_on_rack()
            elif i.startswith("_kp") or i.startswith("_ki"):
                xy1_offset, xy2_offset = self._tensor([0., 0.]), self._tensor([0., 0.])
                while True:
                    obs = self._get_processed_obs()
                    infer = self._inference_keypoints(obs["depth"], obs["clothes"], obs["raw_rgb"])
                    print(f"current predicted logit: {infer['info']['pred_cls']:.4f}")
                    break_flag = False
                    while True:
                        q = input("press c to continue, o / x1, y1, x2, y2 to add offset, otherwise to re-infer\n")
                        if q == "c":
                            break_flag = True
                            break
                        elif q.startswith("o"):
                            result = parse_action(q, 4)
                            if result is None:
                                print(f"invalid action {q}")
                            else:
                                xy1_offset[...] = result[:, :2]
                                xy2_offset[...] = result[:, 2:]
                                break_flag = True
                                break
                        elif q == "r":
                            break_flag = False
                            break
                    if break_flag: break
                if i.startswith("_kp"):
                    self._primitive_keypoints(infer["action"]["left"] + xy1_offset, infer["action"]["right"] + xy2_offset)
                else:
                    self._primitive_keypoints(infer["action"]["right"] + xy2_offset, infer["action"]["left"] + xy1_offset)
            elif i.startswith("_oi"):
                obs = self._get_processed_obs()
                infer = self._inference_keypoints(obs["depth"], obs["clothes"], obs["raw_rgb"])
                print(f"current predicted logit: {infer['info']['pred_cls']:.4f}")
            elif i.startswith("_obs"):
                try:
                    if i.split()[1] == "best":
                        obs = self._get_processed_obs(seg_neckline=True)
                    else:
                        y = float(i.split()[1])
                        obs = self._get_processed_obs(seg_neckline=True, seg_neckline_overwrite=y)
                    neckline = obs["neckline"]
                    print(neckline.sum(), neckline.shape)
                    real_utils.vis.show(neckline)
                except ValueError as e:
                    print(e)
            elif i == "_hanger":
                self._primitive_find_and_pick_hanger()
            elif i == "_e2e":
                self._api_move_to_qpos(self._init_qpos, cache=False)
                while True:
                    '''while True:
                        s = input("press enter to continue, press b to stop end2end inference\n")
                        if s == "":
                            stop_infer = False
                            break
                        elif s == "b":
                            stop_infer = True
                            break
                        else:
                            print(f"invalid input {s}")
                    if stop_infer:
                        break'''
                    obs = self._e2e_get_obs_and_sta()
                    action = self._e2e_predict(**obs)
                    self._e2e_execute(action)
            elif i == "_reset_e2e":
                self._sequence_reset()
                self._e2e_reset()
            elif i == "_infer_all":
                xy1_offset, xy2_offset = self._tensor([0., 0.]), self._tensor([0., 0.])
                obs = self._get_processed_obs()
                infer = self._inference_keypoints(obs["depth"], obs["clothes"], obs["raw_rgb"])
                self._primitive_keypoints(infer["action"]["left"], infer["action"]["right"])
                
                self._primitive_find_and_pick_hanger(skip_query=True)
                obs = self._get_processed_obs(seg_neckline=True)
                infer = self._inference_insert_left(obs["depth"], obs["clothes"], obs["hanger"], obs["neckline"], obs["raw_rgb"])
                xy1, xy2 = self._precheck_lift_press(infer["action"]["action_1_xy"], infer["action"]["action_2_xy"])
                self._primitive_press(xy1)
                self._primitive_lift(xy2)
                
                obs = self._get_processed_obs()
                infer = self._inference_insert_right(obs["depth"], obs["clothes"], obs["hanger"], obs["neckline"], obs["raw_rgb"])
                xy1, xy2 = self._precheck_drag_rotate(infer["action"]["action_1_xy"], infer["action"]["action_2_xy"])
                self._primitive_drag(xy1)
                self._primitive_rotate(xy2)
                self._primitive_put_hanger_on_rack()
            else:
                print(f"unrecognized input {i}")
            return 0
        
        while True:
            try:
                ret = iter_func()
                if ret == 1:
                    break
            except real_utils.SIGUSR1Exception:
                print("catch SIGUSR1 Exception! run reset() ...")
                self._sequence_reset()
            except Exception as e:
                raise e
    
    def _camera_thread(self, frequency):
        cnt = 0
        while self._save_camera_rgb:
            rgb, ir1, ir2 = self._real_api.RSC_HEAD.get_ir_and_rgb_image()
            os.makedirs("./camera", exist_ok=True)
            Image.fromarray(rgb).save(f"./camera/{str(cnt).zfill(6)}.png")
            time_module.sleep(1 / frequency)
            cnt += 1
    
    def _launch_camera_thread(self, launch: False):
        self._launch_camera_thread_state = launch
        if self._launch_camera_thread_state:
            self._save_camera_rgb = True
            self._camera_thread_instance = threading.Thread(target=self._camera_thread, args=(30., ), daemon=True)
            self._camera_thread_instance.start()
        
    def close(self):
        if self._launch_camera_thread_state:
            self._save_camera_rgb = False
            self._camera_thread_instance.join()
