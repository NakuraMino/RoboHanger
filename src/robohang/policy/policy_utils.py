import logging
import taichi as ti

import os
import json
import copy
from typing import Literal, Dict, Callable, Any, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from PIL import Image
import trimesh
import trimesh.transformations as tra

import omegaconf

import robohang.sim.sim_utils as sim_utils
from robohang.sim.cloth import Cloth
import robohang.sim.so3 as so3
import robohang.sim.maths as maths
import robohang.common.utils as utils
from robohang.env.sim_env import SimEnv
from robohang.env.sapien_renderer import CameraProperty, model_matrix_np, camera_pose_to_matrix, camera_property_to_intrinsics_matrix
from robohang.agent.base_agent import BaseAgent

logger = logging.getLogger(__name__)


class ObservationExporter:
    def __init__(
        self, 
        sim_env: SimEnv, 
        agent: BaseAgent, 
        export_cfg: omegaconf.DictConfig, 
        total_traj: int, 
        total_step: int, 
    ) -> None:
        
        assert isinstance(sim_env, SimEnv)
        assert isinstance(agent, BaseAgent)

        self._sim_env = sim_env
        self._agent = agent

        self._cfg = copy.deepcopy(export_cfg)
        self._base_dir = str(self._cfg.base_dir)

        self._side_view_export_mesh = bool(self._cfg.side_view_export_mesh)
        self._side_view_export_mesh_seperately = getattr(self._cfg, "side_view_export_mesh_seperately", False)

        self._side_zfill_len = 6
        self._step_zfill_len = len(str(total_step - 1))
        self._traj_zfill_len = len(str(total_traj - 1))
        self._batch_zfill_len = len(str(self._sim_env.batch_size - 1))

        self._step_idx = 0
        self._dense_step_idx = 0
        self._dense_step_zfill_len = 8
        self._side_idx = 0
        self._traj_idx = 0

    def _get_trajectory_folder_name(self):
        return str(self._traj_idx).zfill(self._traj_zfill_len)

    def get_trajectory_folder_name(self):
        return self._get_trajectory_folder_name()

    def _get_trajectory_path(self):
        """base_dir / traj_idx"""
        return os.path.join(
            self._base_dir, 
            self._get_trajectory_folder_name(),
        )
    
    def get_trajectory_path(self):
        """base_dir / traj_idx"""
        return self._get_trajectory_path()
    
    def _get_step_str(self):
        return str(self._step_idx).zfill(self._step_zfill_len)
    
    def get_step_str(self):
        return self._get_step_str()

    def _get_dense_step_str(self):
        return str(self._dense_step_idx).zfill(self._dense_step_zfill_len)

    def get_dense_step_str(self):
        return self._get_dense_step_str()

    def _get_policy_obs_base_dir(self, batch_idx: int):
        return os.path.join(
            self._get_trajectory_path(),
            "obs", 
            str(batch_idx).zfill(self._batch_zfill_len),
        )
    
    def get_policy_obs_base_dir(self, batch_idx: int):
        return self._get_policy_obs_base_dir(batch_idx)

    def _export_policy_obs(self, batch_idx: int):
        base_dir = self._get_policy_obs_base_dir(batch_idx)
        step_str = self._get_step_str()
        
        result, mask_str_to_idx, camera_info = self._agent.get_obs("direct", "small", batch_idx, True, pos=self._agent.direct_obs)
        logger.info(f"camera_info:{camera_info}")
        os.makedirs(os.path.join(base_dir, "color"), exist_ok=True)
        print('MINOOO base_dir', base_dir)
        Image.fromarray(result["rgba"]).save(os.path.join(base_dir, "color", f"{step_str}.png"))

        os.makedirs(os.path.join(base_dir, "mesh_norobot"), exist_ok=True)
        garment_mesh = self._sim_env.garment.get_mesh(batch_idx)
        trimesh.util.concatenate([
            trimesh.Trimesh(vertices=garment_mesh.vertices, faces=garment_mesh.faces, vertex_colors=self._sim_env.garment_rest_mesh.visual.vertex_colors), 
            self._sim_env.table.get_mesh(batch_idx),
            self._sim_env.grippers["left"].get_mesh(batch_idx),
            self._sim_env.grippers["right"].get_mesh(batch_idx),
            self._sim_env.hanger.get_mesh(batch_idx),
        ]).export(os.path.join(base_dir, "mesh_norobot", f"{step_str}.obj"))

        if self._cfg.obs_export_mesh:
            os.makedirs(os.path.join(base_dir, "mesh"), exist_ok=True)
            self._get_export_mesh(batch_idx).export(os.path.join(base_dir, "mesh", f"{step_str}.obj"))

        os.makedirs(os.path.join(base_dir, "depth"), exist_ok=True)
        np.save(os.path.join(base_dir, "depth", f"{step_str}.npy"), result["depth"])

        os.makedirs(os.path.join(base_dir, "mask"), exist_ok=True)
        np.save(os.path.join(base_dir, "mask", f"{step_str}.npy"), result["mask"])

        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, True, "double_side")
        os.makedirs(os.path.join(base_dir, "reproject_depth"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_depth", f"{step_str}.npy"), reproject_result["depth_output"])

        plt.imshow(reproject_result["depth_output"], cmap="gist_gray")
        plt.colorbar()
        plt.savefig(os.path.join(base_dir, "reproject_depth", f"{step_str}.png"))
        plt.close()

        os.makedirs(os.path.join(base_dir, "reproject_is_garment"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_garment", f"{step_str}.npy"), reproject_result["mask_output"])

        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, True, "inverse_side")
        os.makedirs(os.path.join(base_dir, "reproject_is_inverse"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_inverse", f"{step_str}.npy"), reproject_result["mask_output"])
        
        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, True, "hanger")
        os.makedirs(os.path.join(base_dir, "reproject_is_hanger"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_hanger", f"{step_str}.npy"), reproject_result["mask_output"])

        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, False, "double_side")
        os.makedirs(os.path.join(base_dir, "reproject_is_garment_nointerp"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_garment_nointerp", f"{step_str}.npy"), reproject_result["mask_output"])

        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, False, "inverse_side")
        os.makedirs(os.path.join(base_dir, "reproject_is_inverse_nointerp"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_inverse_nointerp", f"{step_str}.npy"), reproject_result["mask_output"])

        reproject_result, reproject_info = self._agent.get_reproject(result, mask_str_to_idx, camera_info, False, "hanger")
        os.makedirs(os.path.join(base_dir, "reproject_is_hanger_nointerp"), exist_ok=True)
        np.save(os.path.join(base_dir, "reproject_is_hanger_nointerp", f"{step_str}.npy"), reproject_result["mask_output"])

        os.makedirs(os.path.join(base_dir, "info"), exist_ok=True)
        with open(os.path.join(base_dir, "info", f"{step_str}.json"), "w") as f_obj:
            json.dump(
                dict(
                    mask_str_to_idx=mask_str_to_idx, 
                    camera_info=dict(camera_prop=camera_info["camera_prop"].to_dict(), camera_pose=camera_info["camera_pose"]),
                    reproject=reproject_info,
                ), fp=f_obj, indent=4,
            )

    def export_policy_obs(self):
        """
        Export all observation for policy and `self._step_idx += 1`
        """
        for batch_idx in range(self._sim_env.batch_size):
            self._export_policy_obs(batch_idx)
        self._step_idx += 1

    def _export_policy_rgb_only(self, batch_idx: int, subfolder: str = "color_dense", randomize: bool = False):
        """
        Export only the RGB image for the given batch to a dense timeline folder.
        This is lightweight and suitable for per-timestep capture.
        """
        base_dir = self._get_policy_obs_base_dir(batch_idx)
        step_str = self._get_dense_step_str()

        result, _, _ = self._agent.get_obs("direct", "small", batch_idx, randomize, pos=self._agent.direct_obs)
        out_dir = os.path.join(base_dir, subfolder)
        os.makedirs(out_dir, exist_ok=True)
        Image.fromarray(result["rgba"]).save(os.path.join(out_dir, f"{step_str}.png"))

    def callback_rgb_every_timestep(self, env, sim, substep: int):
        """
        Simulation callback: capture RGB at every simulation substep (timestep).
        Increments step index so filenames advance monotonically within a trajectory.
        """
        for batch_idx in range(self._sim_env.batch_size):
            self._export_policy_rgb_only(batch_idx, subfolder="color_dense", randomize=False)
        self._dense_step_idx += 1

    def callback_rgb_every_step_end(self, env, sim, substep: int):
        """
        Simulation callback: capture RGB once per environment step (at end of substeps).
        Useful if you want one frame per step, not per substep.
        """
        if substep == (self._sim_env.sim.substeps - 1):
            for batch_idx in range(self._sim_env.batch_size):
                self._export_policy_rgb_only(batch_idx, subfolder="color_step", randomize=False)
            self._dense_step_idx += 1
    
    def _get_export_mesh(self, batch_idx: int) -> trimesh.Trimesh:
        garment_mesh = self._sim_env.garment.get_mesh(batch_idx)
        return trimesh.util.concatenate([
            trimesh.Trimesh(vertices=garment_mesh.vertices, faces=garment_mesh.faces, vertex_colors=self._sim_env.garment_rest_mesh.visual.vertex_colors), 
            self._sim_env.table.get_mesh(batch_idx),
            self._sim_env.grippers["left"].get_mesh(batch_idx),
            self._sim_env.grippers["right"].get_mesh(batch_idx),
            self._sim_env.robot.get_mesh(batch_idx, use_collision_instead_of_visual=True),
            self._sim_env.hanger.get_mesh(batch_idx),
        ]) 

    def _export_side_view(self, batch_idx: int):
        if hasattr(self._cfg, "side_view_batch_idx"):
            if batch_idx not in self._cfg.side_view_batch_idx:
                return
        
        base_dir = os.path.join(
            self._get_trajectory_path(),
            "side_view", 
            str(batch_idx).zfill(self._batch_zfill_len),
        )
        side_str = str(self._side_idx).zfill(self._side_zfill_len)

        result, mask_str_to_idx, camera_info = self._agent.get_obs("side", "medium", batch_idx, False)
        # logger.info(f"camera_info:{camera_info}")
        os.makedirs(os.path.join(base_dir), exist_ok=True)
        Image.fromarray(result["rgba"]).save(os.path.join(base_dir, f"{side_str}.png"))

        if self._side_view_export_mesh:
            mesh_base_dir = os.path.join(
                self._get_trajectory_path(),
                "mesh", 
                str(batch_idx).zfill(self._batch_zfill_len),
            )

            os.makedirs(mesh_base_dir, exist_ok=True)
            self._get_export_mesh(batch_idx).export(os.path.join(mesh_base_dir, f"{side_str}.obj"))
        
        if self._side_view_export_mesh_seperately:
            mesh_base_dir = os.path.join(
                self._get_trajectory_path(),
                "mesh_sep", str(batch_idx).zfill(self._batch_zfill_len), side_str
            )

            os.makedirs(mesh_base_dir, exist_ok=True)
            self._sim_env.robot.get_mesh(batch_idx, use_collision_instead_of_visual=True).export(os.path.join(mesh_base_dir, "robot.obj"))
            base = utils.torch_to_numpy(so3.pos7d_to_matrix(self._sim_env.robot.get_base_link_pos()[self._sim_env.robot_base_link])[batch_idx, ...]).tolist()
            cfg = {k: float(v[batch_idx]) for k, v in self._sim_env.robot.get_cfg_pos().items()}
            with open(os.path.join(mesh_base_dir, "robot.json"), "w") as f_obj:
                json.dump(dict(
                    base=base, cfg=cfg, 
                ), f_obj, indent=4)

            mesh: trimesh.Trimesh = trimesh.load_mesh(self._sim_env.hanger_vis_path)
            mesh.apply_transform(utils.torch_to_numpy(so3.pos7d_to_matrix(self._sim_env.hanger.get_pos()))[batch_idx, ...])
            mesh.export(os.path.join(mesh_base_dir, "hanger.obj"))

            self._sim_env.garment.get_mesh(batch_idx, vert_norm=True).export(os.path.join(mesh_base_dir, "garment.obj"))

    def callback_side_view(self, env, sim, substep):
        if substep == 0:
            for batch_idx in range(self._sim_env.batch_size):
                self._export_side_view(batch_idx)
            self._side_idx += 1

    def update_traj_idx(self):
        """
        write completed txt
        self._step_idx = 0
        self._side_idx = 0
        self._traj_idx += 1
        """
        with open(os.path.join(self._get_trajectory_path(), "completed.txt"), "w") as f_obj:
            pass
        self._step_idx = 0
        self._dense_step_idx = 0
        self._side_idx = 0
        self._traj_idx += 1


class CameraInfo:
    def __init__(self, camera_prop: dict, camera_pose: list) -> None:
        self.intri = camera_property_to_intrinsics_matrix(CameraProperty(**camera_prop))
        self.extri = camera_pose_to_matrix(camera_pose)


def ijd2xyz(
    i: int,
    j: int,
    d: float,
    camera_info: CameraInfo,
) -> np.ndarray:
    x, y, z = np.linalg.inv(camera_info.intri) @ np.array([j + 0.5, i + 0.5, 1.])
    camera_xyz = np.array([x / z * d, y / z * d, d])
    world_xyz = (camera_info.extri @ model_matrix_np @ np.array([*camera_xyz, 1.]))[:3]
    return world_xyz


def xyz2ij(
    x: float,
    y: float,
    z: float,
    camera_info: CameraInfo,
) -> np.ndarray:
    camera_xyz = (np.linalg.inv(camera_info.extri @ model_matrix_np) @ np.array([x, y, z, 1.]))[:3]
    u, v, w = camera_info.intri @ camera_xyz
    i, j = int(v / w), int(u / w)
    return np.array([i, j])


def action_batch_ij_to_xy(action_center_ij: torch.Tensor, depth: torch.Tensor):
    B, H, W = depth.shape
    dtype, device = depth.dtype, depth.device
    depth_mean_np = utils.torch_to_numpy(depth.view(B, -1).mean(dim=-1)) # [B]

    ij_np = utils.torch_to_numpy(action_center_ij)
    assert ij_np.shape == (B, 2)

    xys = []
    for batch_idx in range(B):
        i, j = ij_np[batch_idx, :]
        x, y, z = ijd2xyz(
            i, j, depth_mean_np[batch_idx], MetaInfo.reproject_camera_info,
        )
        xys.append([x, y])
    
    return torch.tensor(xys, dtype=dtype, device=device)


def action_batch_ij_to_z(action_center_ij: torch.Tensor, depth: torch.Tensor):
    B, H, W = depth.shape
    dtype, device = depth.dtype, depth.device
    depth_mean_np = utils.torch_to_numpy(depth.view(B, -1).mean(dim=-1)) # [B]

    ij_np = utils.torch_to_numpy(action_center_ij)
    assert ij_np.shape == (B, 2)

    zs = []
    for batch_idx in range(B):
        i, j = ij_np[batch_idx, :]
        x, y, z = ijd2xyz(
            i, j, depth_mean_np[batch_idx], MetaInfo.reproject_camera_info,
        )
        zs.append(z)
    
    return torch.tensor(zs, dtype=dtype, device=device)


@ti.kernel
def _calculate_garment_layer_kernel(
    pos: ti.types.ndarray(dtype=ti.math.vec3),
    f2v: ti.types.ndarray(dtype=ti.math.ivec3),
    query_xy: ti.types.ndarray(dtype=ti.math.vec2),
    layer_num: ti.types.ndarray(),
    z_upper_max: ti.types.ndarray(),
    dx_eps: float,
):
    """
    Args:
        pos: [B, V][3], float
        f2v: [F][3], int
        query_xy: [B, Q][2], float
        layer_num: [B, Q], int
        z_upper_max: [B, Q], float
    """
    for batch_idx, qid in ti.ndrange(layer_num.shape[0], layer_num.shape[1]):
        layer_num[batch_idx, qid] = 0
        z_upper_max[batch_idx, qid] = -ti.math.inf
    
    for batch_idx, qid, fid in ti.ndrange(layer_num.shape[0], layer_num.shape[1], f2v.shape[0]):
        xyz = ti.Vector.zero(float, 3)
        xyz[:2] = query_xy[batch_idx, qid]
        a = pos[batch_idx, f2v[fid][0]]
        b = pos[batch_idx, f2v[fid][1]]
        c = pos[batch_idx, f2v[fid][2]]
        uvw = maths.get_3D_barycentric_weights_func(
            xyz, 
            ti.Vector([1., 1., 0.], float) * a, 
            ti.Vector([1., 1., 0.], float) * b, 
            ti.Vector([1., 1., 0.], float) * c, 
            dx_eps,
        )
        u, v, w = uvw
        if (0. <= uvw).all() and (uvw <= 1.).all():
            layer_num[batch_idx, qid] += 1
            ti.atomic_max(
                z_upper_max[batch_idx, qid],
                (a * u + b * v + c * w)[2],
            )


def calculate_garment_layer(
    pos: torch.Tensor,
    f2v: torch.Tensor,
    query_xy: torch.Tensor,
    dx_eps: float,
) -> Dict[str, torch.Tensor]:
    """
    Args:
        pos: [B, V, 3], float
        f2v: [F, 3], int
        query_xy: [B, Q, 2], float
    Return: Dict
        layer_num: [B, Q], int
        z_upper_max: [B, Q], float
    """
    B, V, D = pos.shape
    F, D_ = f2v.shape
    B_, Q, D__ = query_xy.shape
    assert D == D_ == 3 and B == B_ and D__ == 2, f"{pos.shape} {f2v.shape} {query_xy.shape}"

    device = pos.device
    dtype = pos.dtype
    dtype_int= f2v.dtype

    layer_num = torch.zeros((B, Q), dtype=dtype_int, device=device)
    z_upper_max = torch.zeros((B, Q), dtype=dtype, device=device)
    _calculate_garment_layer_kernel(pos.contiguous(), f2v.contiguous(), query_xy.contiguous(), layer_num, z_upper_max, dx_eps)

    return dict(
        layer_num=layer_num,
        z_upper_max=z_upper_max,
    )


@ti.kernel
def _update_iterative_sample_kernel(
    new_sample: ti.types.ndarray(),
    new_is_good: ti.types.ndarray(),
    current_answer: ti.types.ndarray(),
    current_is_good: ti.types.ndarray(),
    update_query: ti.types.ndarray(),
) -> int:
    """
    Args:
        new_sample: [B, Q, D]
        new_is_good: [B, Q]
        current_answer: [B, D]
        current_is_good: [B]
        update_query: [B]
    Return:
        all_is_good: int
    """
    all_is_good = 1

    for batch_idx in range(update_query.shape[0]):
        update_query[batch_idx] = -1
    
    for batch_idx, qid in ti.ndrange(new_is_good.shape[0], new_is_good.shape[1]):
        if new_is_good[batch_idx, qid] and not current_is_good[batch_idx]:
            update_query[batch_idx] = qid
    
    for batch_idx in range(update_query.shape[0]):
        qid = update_query[batch_idx]
        if qid != -1:
            for i in range(current_answer.shape[1]):
                current_answer[batch_idx, i] = new_sample[batch_idx, qid, i]
            current_is_good[batch_idx] = new_is_good[batch_idx, qid]
        if not current_is_good[batch_idx]:
            all_is_good = False
    
    return all_is_good


def iterative_sample(
    generate_sample_func: Callable[[], torch.Tensor],
    is_good_sample_func: Callable[[torch.Tensor], torch.Tensor],
    num_iter: int
):
    """
    Args:
        generate_sample_func: () -> [B, Q, D], float
        is_good_sample_func: ([B, Q, D], float) -> [B, Q], int
    Return:
        current_answer: [B, D], float
        current_is_good: [B, ], int
    """
    assert num_iter >= 1
    
    def generate_sample_func_wrap():
        ans = generate_sample_func().contiguous()
        assert len(ans.shape) == 3, f"{ans.shape}"
        return ans
    
    def is_good_sample_func_wrap(s):
        ans = is_good_sample_func(s).contiguous()
        assert len(ans.shape) == 2, f"{ans.shape}"
        return ans

    _tmp_sample = generate_sample_func_wrap() # [B, Q, D]
    current_answer = _tmp_sample[:, 0, :].contiguous() # [B, D]
    current_is_good = is_good_sample_func_wrap(_tmp_sample)[:, 0].contiguous() # [B, ]
    # logger.info(f"current_is_good:{current_is_good} current_answer:{current_answer}")
    update_query = torch.zeros_like(current_is_good) # [B, ]
    
    for _ in range(num_iter):
        new_sample = generate_sample_func_wrap() # [B, Q, D]
        new_is_good = is_good_sample_func_wrap(new_sample) # [B, Q]
        all_is_good = _update_iterative_sample_kernel(
            new_sample,
            new_is_good,
            current_answer,
            current_is_good,
            update_query,
        )
        # logger.info(f"{_} current_is_good:{current_is_good} current_answer:{current_answer}")
        if all_is_good:
            break
    
    return current_answer, current_is_good


# meta info
class MetaInfo:
    reproject = dict(
        pose = [0., 0.5, 2.5, 0.5, -0.5, 0.5, 0.5],
        prop = dict(height=128, width=128, fx=256., fy=256., cx=64., cy=64., skew=0.,)
    )
    reproject_camera_info = CameraInfo(reproject["prop"], reproject["pose"])
    _H = reproject["prop"]["height"]
    _W = reproject["prop"]["width"]

    _funnel_gym_cfg = omegaconf.OmegaConf.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../config/policy/funnel/funnel_gym.yaml")
    ).funnel_gym

    _insert_gym_cfg = omegaconf.OmegaConf.load(
        os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../config/policy/insert/insert_gym.yaml")
    ).insert_gym

    fling_normal_action_space = _funnel_gym_cfg.primitive.fling.normal.action_space
    pick_place_action_space = _funnel_gym_cfg.primitive.pick_place.action_space
    press_action_space = _insert_gym_cfg.primitive.press.action_space
    lift_action_space = _insert_gym_cfg.primitive.lift.action_space
    drag_action_space = _insert_gym_cfg.primitive.drag.action_space
    rotate_action_space = _insert_gym_cfg.primitive.rotate.action_space

    @classmethod
    def check_reproject_info(self, reproject_info: Dict[Literal["camera_pose", "camera_prop"], Any]):
        if not np.allclose(reproject_info["camera_pose"], self.reproject["pose"]):
            return False
        else:
            for k in self.reproject["prop"].keys():
                if reproject_info["camera_prop"][k] != self.reproject["prop"][k]:
                    return False
        return True
    
    @classmethod
    def get_action_space_slice(self, act_str: str, depth_raw: np.ndarray):
        z = self.calculate_z_world(depth_raw)
        cfg = dict(
            fn=self.fling_normal_action_space,
            pp=self.pick_place_action_space,
            lift=self.lift_action_space,
            press=self.press_action_space,
            drag=self.drag_action_space,
            rotate=self.rotate_action_space,
        )[act_str]
        if act_str in ["lift", "press", "drag", "rotate"]:
            i_max, j_min = xyz2ij(*(cfg.min), z, self.reproject_camera_info)
            i_min, j_max = xyz2ij(*(cfg.max), z, self.reproject_camera_info)
            i_max = max(min(i_max, self._H - 1), 0)
            j_max = max(min(j_max, self._W - 1), 0)
            return slice(i_min, i_max + 1), slice(j_min, j_max + 1)
        elif act_str == "fn":
            ij_center = xyz2ij(*(cfg.center), z, self.reproject_camera_info)
            ij_radius = np.linalg.norm(
                xyz2ij(cfg.center[0] + cfg.radius, cfg.center[1], z, self.reproject_camera_info) -
                xyz2ij(cfg.center[0] - cfg.radius, cfg.center[1], z, self.reproject_camera_info)
            ) / 2
            i, j = np.meshgrid(np.arange(self._H), np.arange(self._W), indexing="ij")
            i, j = np.where((i - ij_center[0]) ** 2 + (j - ij_center[1]) ** 2 < ij_radius ** 2)
            return i, j
        else:
            raise NotImplementedError(act_str)
    
    @classmethod
    def calculate_z_world(self, depth: np.ndarray):
        return self.reproject["pose"][2] - depth.mean()
    
    @classmethod
    def reproject_height(self):
        return self.reproject["prop"]["height"]

    @classmethod
    def reproject_width(self):
        return self.reproject["prop"]["width"]
    
    @classmethod
    def clip_reproject_ij(self, ij: np.ndarray):
        return np.array(ij).clip(min=0, max=[self.reproject_height() - 1, self.reproject_width() - 1])