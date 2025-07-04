import taichi as ti

import logging
logger = logging.getLogger(__name__)

import copy
from typing import Dict, Literal, Optional, List, Callable, Iterable, Tuple, Union
import os
import json
import shutil

import tqdm
import torch
import numpy as np
import pprint

import trimesh
import trimesh.transformations as tra

import omegaconf
import sapien.core as sapien

import robohang.common.utils as utils
import robohang.sim.api as api
from robohang.sim.cloth import Cloth
from robohang.sim.articulate import Articulate
from robohang.sim.actor import ArticulateActor
from robohang.sim.rigid import Rigid, RigidMesh
from robohang.sim.sim import Sim
from robohang.sim.env import Env
import robohang.sim.so3 as so3
import robohang.sim.maths as maths
import robohang.sim.sim_utils as sim_utils
from robohang.env.sapien_renderer import SapienRenderer, CameraProperty


@ti.data_oriented
class RobotGripper:
    ModePick = 0
    ModeHold = 1
    ModeRelease = 2
    def __init__(self, garment: Cloth, robot: Articulate, gripper_mesh: RigidMesh, gripper_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(garment, Cloth)
        assert isinstance(robot, Articulate)
        assert isinstance(gripper_mesh, Rigid)
        assert isinstance(gripper_cfg, omegaconf.DictConfig)

        # objects
        self._garment = garment
        self._gripper_mesh = gripper_mesh
        self._robot = robot

        # constants
        self._nv = self._garment.nv
        self.batch_size = self._garment.batch_size
        self.dtype = self._garment.dtype
        self.dtype_int = self._garment.dtype_int
        self.device = self._garment.device

        # current mode
        self._current_mode = torch.zeros((self.batch_size, ), dtype=self.dtype_int, device=self.device)
        self._current_mode[...] = self.ModeRelease

        # caches
        self._picked_cnt = torch.zeros((self.batch_size, ), dtype=self.dtype_int, device=self.device)
        """int, [B, ]"""
        self._is_picked = torch.zeros((self.batch_size, self._nv), dtype=self.dtype_int, device=self.device)
        """int, [B, V]"""
        self._pick_xyz = torch.zeros((self.batch_size, self._nv, 3), dtype=self.dtype, device=self.device)
        """float, [B, V][3]"""

        # threshold
        # self._th = torch.zeros((self.batch_size, 3), dtype=self.dtype, device=self.device)
        # """float, [B, 3]"""
        # self._th[...] = torch.tensor(gripper_cfg.th, dtype=self.dtype, device=self.device)
        self._th_z = torch.zeros((self.batch_size, ), dtype=self.dtype, device=self.device)
        self._th_z[...] = gripper_cfg.th_z
        self._th_r = torch.zeros((self.batch_size, ), dtype=self.dtype, device=self.device)
        self._th_r[...] = gripper_cfg.th_r

        # stiffness
        self._stiffness = float(gripper_cfg.stiffness)

        # link_str
        self._link_str = None

    @ti.kernel
    def _gripper_kernel(
        self, 
        gripper_pos: ti.types.ndarray(dtype=sim_utils.vec7),
        garment_pos: ti.types.ndarray(dtype=ti.math.vec3),
        external_force: ti.types.ndarray(dtype=ti.math.vec3),
        external_hessian: ti.types.ndarray(dtype=ti.math.mat3), 
        pick_xyz: ti.types.ndarray(dtype=ti.math.vec3),
        is_picked: ti.types.ndarray(dtype=int),
        picked_cnt: ti.types.ndarray(dtype=int),
        th_z: ti.types.ndarray(dtype=float),
        th_r: ti.types.ndarray(dtype=float),
        current_mode: ti.types.ndarray(dtype=int),
    ) -> float:
        avg_err = 0.
        avg_cnt = 0

        for batch_idx in range(self.batch_size):
            picked_cnt[batch_idx] = 0

        for batch_idx, vid in ti.ndrange(self.batch_size, self._nv):
            xyz = garment_pos[batch_idx, vid]

            if current_mode[batch_idx] == self.ModePick:
                xyz_gripper_frame = xyz - gripper_pos[batch_idx][:3]
                if (ti.abs(xyz_gripper_frame[2]) < th_z[batch_idx] and
                    ti.math.length(xyz_gripper_frame[:2]) < th_r[batch_idx]):
                    picked_cnt[batch_idx] += 1
                    is_picked[batch_idx, vid] = 1
                    pick_xyz[batch_idx, vid] = (
                        so3.pos7d_to_matinv_func(gripper_pos[batch_idx]) @ 
                        maths.vec3_pad1_func(xyz)
                    )[:3]

            elif current_mode[batch_idx] == self.ModeHold:
                if (is_picked[batch_idx, vid] == 1):
                    target_xyz = (
                        so3.pos7d_to_matrix_func(gripper_pos[batch_idx]) @ 
                        maths.vec3_pad1_func(pick_xyz[batch_idx, vid])
                    )[:3]
                    external_force[batch_idx, vid] += self._stiffness * (target_xyz - xyz)
                    external_hessian[batch_idx, vid] += self._stiffness * ti.Matrix.identity(dt=float, n=3)

                    avg_err += ti.math.length(target_xyz - xyz)
                    avg_cnt += 1

            elif current_mode[batch_idx] == self.ModeRelease:
                if is_picked[batch_idx, vid] == 1:
                    is_picked[batch_idx, vid] = 0

        return avg_err / avg_cnt

    def get_picked_cnt(self) -> torch.Tensor:
        return self._picked_cnt.clone()
    
    def get_mesh(self, batch_idx: int):
        return self._gripper_mesh.get_mesh(batch_idx)
    
    def get_rigid(self):
        return self._gripper_mesh
    
    def set_link_str(self, link_str: str):
        self._link_str = str(link_str)
    
    @sim_utils.GLOBAL_TIMER.timer
    def callback(self, env: Env, sim: Sim, substep: int):
        assert isinstance(self._link_str, str), f"self._link_str is not str: {self._link_str}"
        gripper_pos = self._robot.get_link_pos(self._link_str)
        gripper_vel = self._robot.get_link_vel(self._link_str)
        
        garment_pos = self._garment.get_pos()
        external_force = self._garment.get_external_force()
        external_hessian = self._garment.get_external_hessian()

        avg_err = self._gripper_kernel(
            gripper_pos, 
            garment_pos,
            external_force, 
            external_hessian, 

            self._pick_xyz, 
            self._is_picked, 
            self._picked_cnt, 

            self._th_z, 
            self._th_r, 
            self._current_mode
        )

        self._garment.set_external_force(external_force)
        self._garment.set_external_hessian(external_hessian)

        logger.info(f"Gripper ModeHold avg_err:{float(avg_err)}")

        pos = gripper_pos.clone()
        vel = gripper_vel.clone()
        pos[:, 3:7] = torch.tensor([1., 0., 0., 0.], dtype=self.dtype, device=self.device)
        vel[:, 3:6] = 0.
        self._gripper_mesh.set_pos(pos)
        self._gripper_mesh.set_vel(vel)

    def set_mode(self, mode: Literal["Pick", "Hold", "Release"]):
        if mode == "Pick":
            self._current_mode[...] = self.ModePick
        elif mode == "Hold":
            self._current_mode[...] = self.ModeHold
        elif mode == "Release":
            self._current_mode[...] = self.ModeRelease
        else:
            raise ValueError(mode)
    
    def set_mode_set_idx(self, mode: Literal["Pick", "Hold", "Release"], batch_idx: int):
        if mode == "Pick":
            self._current_mode[batch_idx] = self.ModePick
        elif mode == "Hold":
            self._current_mode[batch_idx] = self.ModeHold
        elif mode == "Release":
            self._current_mode[batch_idx] = self.ModeRelease
        else:
            raise ValueError(mode)
    
    def get_state(self):
        return dict(
            current_mode=utils.torch_to_numpy(self._current_mode),
            pick_xyz=utils.torch_to_numpy(self._pick_xyz),
            is_picked=utils.torch_to_numpy(self._is_picked),
            picked_cnt=utils.torch_to_numpy(self._picked_cnt),
        )
    
    def set_state(self, state: dict):
        self._current_mode[...] = torch.tensor(state["current_mode"])
        self._pick_xyz[...] = torch.tensor(state["pick_xyz"])
        self._is_picked[...] = torch.tensor(state["is_picked"])
        self._picked_cnt[...] = torch.tensor(state["picked_cnt"])


@ti.data_oriented
class RobotHanger:
    ModeFix = 0
    ModeRelease = 1
    def __init__(self, robot: Articulate, hanger: Rigid) -> None:
        assert isinstance(robot, Articulate)
        assert isinstance(hanger, Rigid)

        self._robot = robot
        self._hanger = hanger

        # constant
        self.batch_size = self._robot.batch_size
        self.dtype = self._robot.dtype
        self.dtype_int = self._robot.dtype_int
        self.device = self._robot.device

        self._link_str = ["none"] * self.batch_size
        self._transform = torch.eye(4, dtype=self.dtype, device=self.device).repeat(self.batch_size, 1, 1) # [B, 4, 4]
        self._current_mode = torch.zeros((self.batch_size, ), dtype=self.dtype_int, device=self.device)
        self._current_mode[...] = self.ModeRelease
    
    @ti.kernel
    def _calculate_hanger_pos_vel_kernel(
        self,
        hanger_pos: ti.types.ndarray(dtype=sim_utils.vec7),
        hanger_vel: ti.types.ndarray(dtype=sim_utils.vec6),
        link_pos: ti.types.ndarray(dtype=sim_utils.vec7),
        link_vel: ti.types.ndarray(dtype=sim_utils.vec6),
        transform: ti.types.ndarray(dtype=ti.math.mat4),
        current_mode: ti.types.ndarray(),
    ):
        for batch_idx in range(self.batch_size):
            if current_mode[batch_idx] == self.ModeFix:
                link_mat = so3.pos7d_to_matrix_func(link_pos[batch_idx])
                hanger_pos[batch_idx] = so3.matrix_to_pos7d_func(link_mat @ transform[batch_idx])

                vel = ti.Vector.zero(float, 6)
                vel[3:] = link_vel[batch_idx][3:]
                vel[:3] = link_vel[batch_idx][:3] + link_vel[batch_idx][3:].cross(
                    hanger_pos[batch_idx][:3] - link_mat[:3, 3]
                )
                hanger_vel[batch_idx] = vel
    
    @sim_utils.GLOBAL_TIMER.timer
    def callback(self, env: Env, sim: Sim, substep: int):
        hanger_pos = self._hanger.get_pos()
        hanger_vel = self._hanger.get_vel()
        for batch_idx in range(self.batch_size):
            if self._current_mode[batch_idx] == self.ModeFix:
                link_pos = self._robot.get_link_pos(self._link_str[batch_idx])
                link_vel = self._robot.get_link_vel(self._link_str[batch_idx])
                self._calculate_hanger_pos_vel_kernel(
                    hanger_pos, hanger_vel,
                    link_pos, link_vel, self._transform, self._current_mode,
                )
        self._hanger.set_pos(hanger_pos)
        self._hanger.set_vel(hanger_vel)

    def set_mode(self, mode: Literal["Fix", "Release"], link_str: str=None, transform: torch.Tensor=None):
        if mode == "Fix":
            assert link_str is not None
            assert transform is not None
            self._current_mode[...] = self.ModeFix
            self._transform[...] = transform
            self._link_str = [str(link_str)] * self.batch_size
        elif mode == "Release":
            self._current_mode[...] = self.ModeRelease
            vel = self._hanger.get_vel()
            vel[...] = 0.
            self._hanger.set_vel(vel)
        else:
            raise ValueError(mode)

    def set_mode_set_idx(self, mode: Literal["Fix", "Release"], batch_idx: int, link_str: str=None, transform: torch.Tensor=None):
        if mode == "Fix":
            assert link_str is not None
            assert transform is not None
            self._current_mode[batch_idx] = self.ModeFix
            self._transform[batch_idx, :, :] = transform
            self._link_str[batch_idx] = str(link_str)
        elif mode == "Release":
            self._current_mode[batch_idx] = self.ModeRelease
            vel = self._hanger.get_vel()
            vel[batch_idx, :] = 0.
            self._hanger.set_vel(vel)
        else:
            raise ValueError(mode)
    
    def get_state(self):
        return dict(
            current_mode=utils.torch_to_numpy(self._current_mode),
            transform=utils.torch_to_numpy(self._transform),
            link_str=copy.deepcopy(self._link_str),
        )
    
    def set_state(self, state: dict):
        self._current_mode[...] = torch.tensor(state["current_mode"])
        self._transform[...] = torch.tensor(state["transform"])
        if isinstance(state["link_str"], str):
            self._link_str = [state["link_str"]] * self.batch_size
        elif isinstance(state["link_str"], list):
            self._link_str = state["link_str"]
        else:
            raise ValueError(state["link_str"])


class QposInterpolator:
    def __init__(self, artic: Articulate, actor: ArticulateActor, dt: float) -> None:
        assert isinstance(artic, Articulate)
        assert isinstance(actor, ArticulateActor)
        assert isinstance(dt, float)

        self._artic = artic
        self._actor = actor
        self._dt = float(dt)

        self._remain_substeps = 0
        self._total_substeps = 0
        self._curr_qpos = dict()
        self._goal_qpos = dict()
    
    def set_goal(self, total_substeps: int):
        assert total_substeps >= 1
        self._remain_substeps = int(total_substeps)
        self._total_substeps = int(total_substeps)
        self._curr_qpos = dict()
        self._goal_qpos = dict()

        curr_cfg_pos = self._artic.get_cfg_pos()
        goal_qpos = self._actor.get_target()["pos"]
        for joint_name in goal_qpos.keys():
            self._curr_qpos[joint_name] = curr_cfg_pos[joint_name].clone()
            self._goal_qpos[joint_name] = goal_qpos[joint_name].clone()
    
    def callback(self, *args, **kwargs):
        if self._remain_substeps > 0:
            self._remain_substeps -= 1
            cfg_pos = dict()
            cfg_vel = dict()
            for joint_name in self._curr_qpos.keys():
                cfg_pos[joint_name] = (
                    self._remain_substeps * self._curr_qpos[joint_name] + 
                    (self._total_substeps - self._remain_substeps) * self._goal_qpos[joint_name]
                ) / self._total_substeps
                if self._remain_substeps > 0:
                    cfg_vel[joint_name] = (
                        self._goal_qpos[joint_name] - self._curr_qpos[joint_name]
                    ) / self._total_substeps / self._dt
                else:
                    cfg_vel[joint_name] = 0.
            self._artic.set_cfg_pos_vel(cfg_pos=cfg_pos, cfg_vel=cfg_vel)


@ti.data_oriented
class PenetrationChecker:
    def __init__(
        self, 
        sim_env: 'SimEnv',
        penetration_cfg: omegaconf.DictConfig,
    ) -> None:
        assert isinstance(sim_env, SimEnv)
        assert isinstance(penetration_cfg, omegaconf.DictConfig)
        self._sim_env = sim_env
        self._cfg = copy.deepcopy(penetration_cfg)

        self.batch_size = self._sim_env.batch_size
        self.device = self._sim_env.device

        self._penetrate_status: Dict[str, torch.Tensor] = dict(
            garment_self=torch.zeros((self.batch_size, ), dtype=bool, device=self.device),
            garment_robot=torch.zeros((self.batch_size, ), dtype=bool, device=self.device),
            garment_hanger=torch.zeros((self.batch_size, ), dtype=bool, device=self.device),
            gripper_hanger_left=torch.zeros((self.batch_size, ), dtype=bool, device=self.device),
            gripper_hanger_right=torch.zeros((self.batch_size, ), dtype=bool, device=self.device),
        )
        self._check_hanger_gripper: Dict[str, bool] = dict(left=False, right=False)

    @sim_utils.GLOBAL_TIMER.timer
    def callback(self, *args, **kwargs):
        # garment self intersection
        si = self._sim_env.sim.detect_cloth_self_intersection(self._sim_env.garment.name).to(dtype=bool)
        self._penetrate_status["garment_self"] |= si

        # garment robot
        for v in self._sim_env.collision_garment_robot.collision_map.values():
            self._penetrate_status["garment_robot"] |= v.calculate_penetration(self._cfg.garment_robot_tolerance_sdf).to(dtype=bool)
        
        # garment hanger
        self._penetrate_status["garment_hanger"] |= self._sim_env.collision_garment_hanger.calculate_penetration(self._cfg.garment_hanger_tolerance_sdf).to(dtype=bool)
        
        # gripper hanger
        def set_z(r: Rigid):
            pos_old = r.get_pos()
            pos_new = pos_old.clone()
            pos_new[:, 2] = self._sim_env.get_table_height() # only consider intersection on xy-plane
            r.set_pos(pos_new)
            return pos_old
        for h in ["left", "right"]:
            if not self._check_hanger_gripper[h]:
                continue
            gripper_pos_old = set_z(self._sim_env.grippers[h].get_rigid())
            hanger_pos_old = set_z(self._sim_env.hanger)

            self._penetrate_status[f"gripper_hanger_{h}"] |= \
                self._sim_env.hanger.check_other_surface_penetrate_self_rigid(
                    self._sim_env.grippers[h].get_rigid(), 
                    self._cfg.hanger_gripper_tolerance_sdf,
                ).to(dtype=bool)
            
            self._sim_env.grippers[h].get_rigid().set_pos(gripper_pos_old)
            self._sim_env.hanger.set_pos(hanger_pos_old)
            
    def set_check_gripper_hanger(self, left: bool, right: bool):
        self._check_hanger_gripper["left"] = bool(left)
        self._check_hanger_gripper["right"] = bool(right)
        
    def reset(self):
        for v in self._penetrate_status.values():
            v[...] = False

    def get_penetrate_status(self) -> Dict[str, torch.Tensor]:
        """str -> bool, [B, ]"""
        return {k: v.clone() for k, v in self._penetrate_status.items()}
    
    def set_penetrate_status(self, penetrate_status: Dict[str, torch.Tensor]):
        """str -> bool, [B, ]"""
        for k in self._penetrate_status.keys():
            self._penetrate_status[k][...] = torch.tensor(penetrate_status[k])


@ti.data_oriented
class SimEnv:
    def __init__(self, sim_env_cfg: omegaconf.DictConfig, glb_cfg: omegaconf.DictConfig) -> None:
        assert isinstance(sim_env_cfg, omegaconf.DictConfig)
        assert isinstance(glb_cfg, omegaconf.DictConfig)
        
        self._cfg = copy.deepcopy(sim_env_cfg)
        self._glb_cfg = copy.deepcopy(glb_cfg)

        # check usage
        _tmp_path = utils.get_path_handler()(self._cfg.misc.tmp_path)
        self._tmp_path = os.path.join(_tmp_path, str(os.getpid()))

        os.makedirs(self._tmp_path, exist_ok=False)
        print(f"current tmp_path: {_tmp_path}\nsize: {np.round(utils.get_folder_size(_tmp_path) / (2 ** 20))}M")
        print(f"init gym ...")

        # init gym
        self._gym = api.acquire_gym()
        self._sim = self._gym.create_sim(self._cfg.sim, self._glb_cfg)
        self._env = self._gym.create_env(self._sim, self._glb_cfg)

        # create table
        table_path = utils.get_path_handler()(self._cfg.asset.table.mesh_path)
        table_mesh = trimesh.load_mesh(table_path)
        self._table = self._gym.create_rigid(self._sim, self._cfg.asset.table.cfg, self._glb_cfg, mesh=table_mesh)
        self._table.set_pos(torch.tensor([self._cfg.asset.table.pos] * self.batch_size))

        # create garment
        garment_path = utils.get_path_handler()(self._cfg.asset.garment.mesh_path)
        self._garment_mesh = trimesh.load_mesh(garment_path)
        self._garment = self._gym.create_cloth(self._sim, self._garment_mesh, self._cfg.asset.garment.cfg, self._glb_cfg)
        self._garment.set_pos(self._garment.get_pos() + torch.tensor(self._cfg.asset.garment.translation, dtype=self.dtype, device=self.device))
        self._garment_meta = omegaconf.OmegaConf.load(garment_path + ".meta.yaml")

        # create hanger
        hanger_path = utils.get_path_handler()(self._cfg.asset.hanger.mesh_path)
        self._hanger_vis_path = utils.get_path_handler()(self._cfg.asset.hanger.mesh_vis_path)
        hanger_mesh = trimesh.load_mesh(hanger_path)
        self._hanger = self._gym.create_rigid(self._sim, self._cfg.asset.hanger.cfg, self._glb_cfg, mesh=hanger_mesh)
        self._hanger.set_pos(torch.tensor([self._cfg.asset.hanger.pos] * self.batch_size))
        self._hanger_meta = omegaconf.OmegaConf.load(hanger_path + ".meta.yaml")

        # create robot
        self._robot = self._gym.create_articulate(self._sim, self._cfg.asset.robot.cfg, self._glb_cfg)
        self._robot_base_link = str(self._cfg.asset.robot.base_link)
        qpos, qvel = self._robot.get_cfg_pos(), self._robot.get_cfg_vel()
        self._robot_zero_qpos = {k: v.clone() for k, v in qpos.items()}
        self._robot_zero_qvel = {k: v.clone() for k, v in qvel.items()}

        # create actor
        self._actor = self._gym.create_articulate_actor(self._env, self._robot, self._cfg.actor.robot.cfg, self._glb_cfg)
        actor_prop = self._actor.get_actor_dof_properties()

        self._default_actor_speed: Dict[Literal["veryfast", "fast", "medium", "slow", "veryslow"], dict] = dict()
        for speed_str in ["veryfast", "fast", "medium", "slow", "veryslow"]:
            prop = dict()
            for k, v in actor_prop.items():
                prop[k] = dict()
                # direct copy
                for kk, vv in v.items():
                    if isinstance(vv, torch.Tensor):
                        prop[k][kk] = vv.clone()
                    elif isinstance(vv, (float, type(None), str)):
                        prop[k][kk] = vv
                    else:
                        raise TypeError(vv)
                
                # set value
                prop[k]["driveMode"] = "PDAcceleration"
                if speed_str == "veryfast":
                    prop[k]["stiffness"][...] = 900.
                    prop[k]["damping"][...] = 60.
                    prop[k]["vel_limit"] = 100.0
                elif speed_str == "fast":
                    prop[k]["stiffness"][...] = 900.
                    prop[k]["damping"][...] = 60.
                    prop[k]["vel_limit"] = 3.0
                elif speed_str == "medium":
                    prop[k]["stiffness"][...] = 400.
                    prop[k]["damping"][...] = 40.
                    prop[k]["vel_limit"] = 2.0
                elif speed_str == "slow":
                    prop[k]["stiffness"][...] = 196.
                    prop[k]["damping"][...] = 28.
                    prop[k]["vel_limit"] = 1.4
                elif speed_str == "veryslow":
                    prop[k]["stiffness"][...] = 64.
                    prop[k]["damping"][...] = 16.
                    prop[k]["vel_limit"] = 0.8
                else:
                    raise NotImplementedError(speed_str)
            self._default_actor_speed[speed_str] = prop

        logger.info(f"self._default_actor_speed:\n{pprint.pformat(self._default_actor_speed, sort_dicts=False)}")
        self._actor.set_actor_dof_properties(self._default_actor_speed["medium"])

        # add gripper
        self._grippers_mesh: Dict[Literal["left", "right"], RigidMesh] = dict(
            left=self._gym.create_rigid(
                self._sim, 
                self._cfg.asset.grippers.left_cfg, 
                self._glb_cfg, 
                mesh=trimesh.primitives.Cylinder(**(self._cfg.asset.grippers.prop)), 
            ),
            right=self._gym.create_rigid(
                self._sim, 
                self._cfg.asset.grippers.right_cfg, 
                self._glb_cfg, 
                mesh=trimesh.primitives.Cylinder(**(self._cfg.asset.grippers.prop)), 
            ),
        )

        self._grippers: Dict[Literal["left", "right"], RobotGripper] = dict(
            left=RobotGripper(
                self._garment, 
                self._robot, 
                self._grippers_mesh["left"], 
                self._cfg.asset.grippers.parameter, 
            ), 
            right=RobotGripper(
                self._garment, 
                self._robot, 
                self._grippers_mesh["right"], 
                self._cfg.asset.grippers.parameter, 
            ),
        )

        # add robot hanger
        self._robot_hanger = RobotHanger(self._robot, self._hanger)

        # add collision
        self._garment_self = self._gym.add_cloth_self_force_collision(
            self._sim, self._garment, 
            self._cfg.sim.garment_self_collision, self._glb_cfg
        )
        self._garment_table = self._gym.add_cloth_rigid_position_collision(
            self._sim, self._garment, self._table, 
            self._cfg.sim.garment_table, self._glb_cfg
        )
        self._garment_hanger = self._gym.add_cloth_rigid_force_collision(
            self._sim, self._garment, self._hanger, 
            self._cfg.sim.garment_hanger, self._glb_cfg
        )
        self._garment_robot = self._gym.add_cloth_articulate_force_collision(
            self._sim, self._garment, self._robot, 
            self._cfg.sim.garment_robot, self._glb_cfg
        )

        # add renderer
        self._renderer = SapienRenderer(
            utils.get_path_handler()(self._cfg.asset.robot.cfg.urdf_path),
            table_path, self._hanger_vis_path, 
        )

        # create helper
        self._qpos_interpolator = QposInterpolator(self._robot, self._actor, self._sim.dt)
        self._penetration_checker = PenetrationChecker(self, self._cfg.parameter.penetration)

        # misc
        self._use_tqdm = bool(self._cfg.misc.use_tqdm)
        
        if self._cfg.misc.enable_timer:
            sim_utils.GLOBAL_TIMER.set_enable()
        else:
            sim_utils.GLOBAL_TIMER.set_disable()

        self._debugger = dict(prev_state=None, curr_state=None)

        # callbacks
        self._callbacks = []

        if self._cfg.misc.enable_debugger:
            self._callbacks.append(self._debugger_callback)
        self._callbacks.append(self._set_fatal_batch_garment_to_init_callback)

        self._callbacks.append(self._robot_hanger.callback)
        self._callbacks.append(self._qpos_interpolator.callback)
        self._callbacks.append(self._grippers["left"].callback)
        self._callbacks.append(self._grippers["right"].callback)
        self._callbacks.append(self._penetration_checker.callback)

        # record current state
        self._init_state = self._get_state()
        self._garment_init_pos = self._garment.get_pos()
        self._garment_init_vel = self._garment.get_vel()

    def _debugger_callback(self, *args, **kwargs):
        self._debugger["prev_state"] = self._debugger["curr_state"]
        self._debugger["curr_state"] = self._get_state()

    def _set_fatal_batch_garment_to_init_callback(self, *args, **kwargs):
        fatal_batches = torch.where(self._sim.get_global_fatal_flag())[0]
        pos = self._garment.get_pos()
        vel = self._garment.get_vel()
        pos[fatal_batches, ...] = self._garment_init_pos[fatal_batches, ...]
        vel[fatal_batches, ...] = self._garment_init_vel[fatal_batches, ...]
        self._garment.set_pos(pos)
        self._garment.set_vel(vel)

    def _tqdm_wrapper(self, r: Iterable) -> Iterable:
        return r if not self._use_tqdm else tqdm.tqdm(r)

    @property
    def batch_size(self):
        return self._sim.batch_size
    
    @property
    def device(self):
        return self._sim.device
    
    @property
    def dtype(self):
        return self._sim.dtype
    
    @property
    def dtype_int(self):
        return self._sim.dtype_int

    @property
    def table(self):
        return self._table

    @property
    def hanger(self):
        return self._hanger
    
    @property
    def hanger_vis_path(self):
        return self._hanger_vis_path
    
    @property
    def hanger_meta(self):
        return self._hanger_meta
    
    @property
    def garment(self):
        return self._garment
    
    @property
    def garment_keypoints(self) -> Dict[str, int]:
        return omegaconf.OmegaConf.to_container(self._garment_meta.key_points)
    
    @property
    def garment_rest_mesh(self) -> trimesh.Trimesh:
        return self._garment_mesh

    @property
    def robot(self):
        return self._robot

    @property
    def actor(self):
        return self._actor
    
    @property
    def robot_base_link(self):
        return self._robot_base_link
    
    @property
    def grippers(self):
        return self._grippers
    
    @property
    def robot_hanger(self):
        return self._robot_hanger
    
    @property
    def collision_garment_self(self):
        return self._garment_self
    
    @property
    def collision_garment_table(self):
        return self._garment_table
    
    @property
    def collision_garment_robot(self):
        return self._garment_robot
    
    @property
    def collision_garment_hanger(self):
        return self._garment_hanger
    
    @property
    def penetration_checker(self):
        return self._penetration_checker
    
    @property
    def gym(self):
        return self._gym

    @property
    def env(self):
        return self._env
    
    @property
    def sim(self):
        return self._sim

    @property
    def renderer(self):
        return self._renderer
    
    @property
    def robot_zero_qpos(self):
        return self._robot_zero_qpos
    
    @property
    def robot_zero_qvel(self):
        return self._robot_zero_qvel
    
    @property
    def tmp_path(self):
        return self._tmp_path
    
    def _tensor(self, x):
        return torch.tensor(np.array(x), dtype=self.dtype, device=self.device)
    
    def domain_randomize(self):
        dr_cfg = self._cfg.parameter.dr
        def randomize(f: ti.Field, name: str):
            val = utils.map_01_ab(
                torch.rand((self.batch_size, ), dtype=self.dtype, device=self.device), 
                getattr(dr_cfg, name)[0], getattr(dr_cfg, name)[1],
            )
            f.from_torch(val)
            logger.info(f"domain randomize {name}:\n{val}")
            
        randomize(self._garment._E, "garment_E")
        randomize(self._garment._rho, "garment_rho")
        randomize(self._garment_self._mu, "garment_self_mu")
        randomize(self._garment_table._mu, "garment_table_mu")
        randomize(self._garment_hanger._mu, "garment_hanger_mu")
        
    def _get_robot_state_dict(self, batch_idx: int):
        state = dict(
            base_mat=utils.torch_to_numpy(
                so3.pos7d_to_matrix(self._robot.get_base_link_pos()[self._robot_base_link])[batch_idx, ...]
            ).tolist(),
            pos_cfg={
                k: utils.torch_to_numpy(v)[batch_idx, ...].tolist()
                for k, v in self._robot.get_cfg_pos().items()
            },
        )
        return state
    
    def get_robot_state_dict(self, batch_idx: int):
        return self._get_robot_state_dict(batch_idx)

    def take_picture(self, camera_prop: CameraProperty, camera_pose: List[float], batch_idx: int):
        def invert_face(m: trimesh.Trimesh):
            m.invert()
            return m
        # save obj to tmp_path
        table_path = os.path.join(self._tmp_path, "table.npy")
        hanger_path = os.path.join(self._tmp_path, "hanger.npy")
        garment_path = os.path.join(self._tmp_path, "garment.obj")
        garment_inv_path = os.path.join(self._tmp_path, "garment_inv.obj")
        robot_path = os.path.join(self._tmp_path, "robot.obj")
        gripper_l_path = os.path.join(self._tmp_path, "gripper_l.obj")
        gripper_r_path = os.path.join(self._tmp_path, "gripper_r.obj")

        table_pose = utils.torch_to_numpy(self._table.get_pos()[batch_idx])
        np.save(table_path, tra.translation_matrix(table_pose[0:3]) @ tra.quaternion_matrix(table_pose[3:7]))
        hanger_pose = utils.torch_to_numpy(self._hanger.get_pos()[batch_idx])
        np.save(hanger_path, tra.translation_matrix(hanger_pose[0:3]) @ tra.quaternion_matrix(hanger_pose[3:7]))
        self._garment.get_mesh(batch_idx, vert_norm=True).export(garment_path)
        invert_face(self._garment.get_mesh(batch_idx, vert_norm=True)).export(garment_inv_path)
        with open(robot_path, "w") as f_obj:
            json.dump(self._get_robot_state_dict(batch_idx), f_obj, indent=4)
        self._grippers["left"].get_mesh(batch_idx).export(gripper_l_path)
        self._grippers["right"].get_mesh(batch_idx).export(gripper_r_path)
        
        # render
        mask_str_to_idx = self._renderer.set_scene(
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
        
        return self._renderer.render(), mask_str_to_idx

    def set_actor_speed(self, speed: Literal["veryfast", "fast", "medium", "slow", "veryslow", "interp"], steps=None):
        logger.info(f"set_actor_speed:{speed} {steps}")
        if speed in ["veryfast", "fast", "medium", "slow", "veryslow"]:
            self._actor.set_actor_dof_properties(self._default_actor_speed[speed])
        elif speed == "interp":
            assert isinstance(steps, int)
            self._qpos_interpolator.set_goal(steps * self._sim.substeps)
        else:
            raise NotImplementedError(speed)
    
    def set_substep(self, mode: Literal["efficient", "accurate", "superreal"]):
        if mode == "efficient":
            self._sim.set_num_substeps(self._cfg.parameter.substeps.efficient_substeps)
        elif mode == "accurate":
            self._sim.set_num_substeps(self._cfg.parameter.substeps.accurate_substeps)
        elif mode == "superreal":
            self._sim.set_num_substeps(self._cfg.parameter.substeps.superreal_substeps)
        else:
            raise ValueError(mode)

    def get_table_height(self):
        """[B, ]"""
        return self._table.get_pos()[:, 2]
        
    def simulate(self, steps: int, callbacks: Optional[List[Callable]]=None):
        assert isinstance(steps, int)
        if callbacks is not None:
            assert isinstance(callbacks, list)
            for callback in callbacks:
                assert callable(callback)
        else:
            callbacks = []

        for i in self._tqdm_wrapper(range(steps)):
            self._gym.simulate(self._env, self._sim, self._callbacks + callbacks)

    def get_penetrate_status(self):
        return self._penetration_checker.get_penetrate_status()
    
    def _get_sim_error(self):
        penetrate_status = self._penetration_checker.get_penetrate_status()
        d = dict(
            all=[False] * self.batch_size,
            global_fatal_flag=sim_utils.torch_to_numpy(self.sim.get_global_fatal_flag()).astype(dtype=bool).tolist(),
        )
        for k, v in penetrate_status.items():
            d[k] = sim_utils.torch_to_numpy(v).astype(dtype=bool).tolist()
        d["all"] = utils.list_or(*[v for v in d.values()])
        return d
    
    def get_sim_error(self):
        return self._get_sim_error()
        
    def _reset(self):
        self._set_state(self._init_state)

    def reset(self):
        self._reset()

    def _get_state(self):
        return dict(
            sim=self._sim.get_state(), 
            env=self._env.get_state(),
            penetration_checker=utils.torch_dict_to_numpy_dict(
                self._penetration_checker.get_penetrate_status()
            ),
            gripper=dict(
                left=self._grippers["left"].get_state(),
                right=self._grippers["right"].get_state(),
            ),
            robot_hanger=self._robot_hanger.get_state(),
        )
    
    def get_state(self):
        return self._get_state()
    
    def _set_state(self, state):
        self._sim.set_state(state["sim"])
        self._env.set_state(state["env"])
        self._penetration_checker.set_penetrate_status(state["penetration_checker"])
        self._grippers["left"].set_state(state["gripper"]["left"])
        self._grippers["right"].set_state(state["gripper"]["right"])
        self._robot_hanger.set_state(state["robot_hanger"])

    def set_state(self, state: dict):
        self._set_state(state)
    
    def __del__(self):
        print("delete sim_env ...")
        shutil.rmtree(self._tmp_path)