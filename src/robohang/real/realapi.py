from typing import Dict, Literal, Optional, List
from typing import get_args as typing_get_args
from PIL import Image
import matplotlib.pyplot as plt
import time as time_module
import imageio
import pickle
import cv2
import pprint

import taichi as ti

import numpy as np
import trimesh.transformations as tra
import scipy.interpolate as interp
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix

import robohang.real.utils as real_utils

import omegaconf
import sys, os
import copy

CONTROLLABLE_JOINT_LIST_TYPE = Literal[
    "leg_joint1",
    "leg_joint2",
    "leg_joint3",
    "leg_joint4",

    "left_arm_joint1",
    "left_arm_joint2",
    "left_arm_joint3",
    "left_arm_joint4",
    "left_arm_joint5",
    "left_arm_joint6",
    "left_arm_joint7",

    "right_arm_joint1",
    "right_arm_joint2",
    "right_arm_joint3",
    "right_arm_joint4",
    "right_arm_joint5",
    "right_arm_joint6",
    "right_arm_joint7",
] # Do Not Modify This !
CONTROLLABLE_JOINT_LIST = [_ for _ in typing_get_args(CONTROLLABLE_JOINT_LIST_TYPE)]


@ti.kernel
def _make_matrix_kernel(
    output_is_hole: ti.types.ndarray(dtype=int),
    row: ti.types.ndarray(dtype=int),
    col: ti.types.ndarray(dtype=int),
    val: ti.types.ndarray(dtype=float),

    depth_output: ti.types.ndarray(dtype=float),
    depth_rhs: ti.types.ndarray(dtype=float),
) -> int:
    triplet_cnt = 0
    u = ti.Vector([0, +1, 0, -1, 0], dt=int)
    v = ti.Vector([0, 0, +1, 0, -1], dt=int)

    for i, j in ti.ndrange(depth_output.shape[0], depth_output.shape[1]):
        ij = i * depth_output.shape[1] + j
        if output_is_hole[i, j] == 1:
            depth_rhs[ij] = 0.

            neighbor_cnt = 0.
            for k in ti.static(range(u.get_shape()[0])):
                ii = i + u[k]
                jj = j + v[k]
                if (
                    (0 <= ii and ii < depth_output.shape[0]) and
                    (0 <= jj and jj < depth_output.shape[1])
                ):
                    if k != 0:
                        neighbor_cnt += 1.
            
            for k in ti.static(range(u.get_shape()[0])):
                ii = i + u[k]
                jj = j + v[k]
                if (
                    (0 <= ii and ii < depth_output.shape[0]) and
                    (0 <= jj and jj < depth_output.shape[1])
                ):
                    iijj = ii * depth_output.shape[1] + jj
                    tid = ti.atomic_add(triplet_cnt, +1)
                    row[tid] = ij
                    col[tid] = iijj
                    if k == 0:
                        val[tid] = neighbor_cnt
                    else:
                        val[tid] = -1.
        else:
            depth_rhs[ij] = depth_output[i, j]

            tid = ti.atomic_add(triplet_cnt, +1)
            col[tid] = ij
            row[tid] = ij
            val[tid] = 1.
                        
    return triplet_cnt


def _preprocess_depth(depth: np.ndarray) -> np.ndarray:
    depth_rhs = np.zeros_like(depth, dtype=np.float32).flatten()
    row = np.zeros(depth_rhs.shape[0] * 5, dtype=np.int32)
    col = np.zeros(depth_rhs.shape[0] * 5, dtype=np.int32)
    val = np.zeros(depth_rhs.shape[0] * 5, dtype=np.float32)
    output_is_hole = (depth == 0).astype(np.int32)
    triplet_cnt = _make_matrix_kernel(output_is_hole, row, col, val, depth, depth_rhs)
    mat = csc_matrix((val[:triplet_cnt], (row[:triplet_cnt], col[:triplet_cnt])), shape=(depth_rhs.shape[0], depth_rhs.shape[0]))
    depth_lhs = splinalg.spsolve(mat, depth_rhs)
    depth_output = depth_lhs.reshape(depth.shape) / 1000.
    return depth_output


try:
    file_dir = os.path.dirname(os.path.abspath(__file__))
    lib_dir = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../../../../"))
    src_dir = os.path.join(lib_dir, "src")
    sys.path.append(src_dir)
    from moma_segmentation.grounded_segment import GroundedSegmentation
    import robohang.real.sam_util as sam_util
    from robohang.real.gsocket_utils import GalbotClient

except ImportError as e:
    print(e)


class RealAPI:
    def __init__(self, cfg: omegaconf.DictConfig):
        print(cfg)
        self._single_step = bool(cfg.single_step)
        self._fast_pass = bool(cfg.fast_pass)
        # print(cfg.hello.ip)
        
        self.head_joints = [0., np.deg2rad(20)]
        
        if not self._fast_pass:
            print("init robot control ...")
            self.RC = GalbotClient()
            print("init grounded sam ...")
            self.GS = self._load_gs(os.path.join(file_dir, "cache", "gs", "cache.pkl"))
            print("init sam ...")
            self.SAM = sam_util.SamModel()
            print("hardware init done ...")
            qpos = self.RC.get_qpos()
            qpos[4:6] = self.head_joints
            self.RC.set_qpos(qpos)
    
    @staticmethod
    def _load_gs(GS_CACHE_PATH):
        os.makedirs(os.path.dirname(GS_CACHE_PATH), exist_ok=True)
        if not os.path.exists(GS_CACHE_PATH):
            GS = GroundedSegmentation()
            with open(GS_CACHE_PATH, "wb") as f_obj:
                pickle.dump(GS, f_obj)
        else:
            with open(GS_CACHE_PATH, "rb") as f_obj:
                GS = pickle.load(f_obj)
        GS.threshold = 0.2
        return GS

    def move_to_qpos(self, qpos: Dict[CONTROLLABLE_JOINT_LIST_TYPE, float], time: Optional[float]=None, overwrite_single_step=None) -> None:
        """
        Move to `qpos` from `current_qpos`. 
        - If `time` is `None`, use the default speed. 
        - If `time` is `float`, use `time` seconds to reach the target qpos. 
        """
        if self._fast_pass:
            return
        
        single_step = self._single_step if overwrite_single_step is None else overwrite_single_step
        if single_step:
            input(f"press enter to move the arm:\n{pprint.pformat(qpos, sort_dicts=False, compact=True)}\n")
        
        qpos_np = self.RC.get_qpos()
        qpos_np[0:4] = np.array([qpos[f"leg_joint{i}"] for i in [1, 2, 3, 4]])
        qpos_np[6:13] = np.array([qpos[f"left_arm_joint{i}"] for i in [1, 2, 3, 4, 5, 6, 7]])
        qpos_np[13:20] = np.array([qpos[f"right_arm_joint{i}"] for i in [1, 2, 3, 4, 5, 6, 7]])
        self.RC.set_qpos(qpos_np, speed=0.5)
    
    def move_to_qpos_list(
        self, 
        qpos_list: List[Dict[CONTROLLABLE_JOINT_LIST_TYPE, float]], 
        joint_limit: List[Dict[CONTROLLABLE_JOINT_LIST_TYPE, List[float]]],
        time_list: List[float],
        frequency=50,
    ) -> None:
        if self._fast_pass:
            return
        
        self.move_to_qpos(qpos={k: v for k, v in qpos_list[0].items()}, time=None, overwrite_single_step=False)
        if self._single_step:
            input(f"press enter to move the arm:\n{pprint.pformat(qpos_list, sort_dicts=False, compact=True)}\n")
        
        n = len(qpos_list)
        interp_result = {
            "leg": {f"leg_joint{i}": None for i in range(1, 5)},
            "left_arm": {f"left_arm_joint{i}": None for i in range(1, 8)},
            "right_arm": {f"right_arm_joint{i}": None for i in range(1, 8)},
        }

        assert time_list[0] == 0., f"time_list:{time_list} should start from 0."
        x = np.cumsum(time_list).tolist()
        xmax = x[-1]
        total_interp_num = round(xmax * frequency) + 1
        for hardware in interp_result.keys():
            for joint in interp_result[hardware]:
                y = [qpos_list[i][joint] for i in range(n)]
                interp_result[hardware][joint] = interp.Akima1DInterpolator(
                    [-2., -1.] + x + [xmax + 1., xmax + 2.],
                    [y[0]] * 2 + y + [y[-1]] * 2,
                )(np.linspace(0, xmax, total_interp_num)) # more smooth interpolation
        pos = {
            "leg": [[] for _ in range(total_interp_num)],
            "left_arm": [[] for _ in range(total_interp_num)],
            "right_arm": [[] for _ in range(total_interp_num)],
        }
        for hardware in interp_result.keys():
            for i in range(total_interp_num):
                pos[hardware][i] = [
                    np.clip(interp_result[hardware][joint][i], joint_limit[joint][0], joint_limit[joint][1])
                    for joint in interp_result[hardware].keys()
                ]
        self.RC.follow_trajectory_mul(
            ["right_arm", "left_arm", "leg"],
            [pos["right_arm"], pos["left_arm"], pos["leg"]],
            asynchronous=True,
        )
        # self.RC.control_interface.follow_trajectory({"positions":pos["leg"]}, "leg", asynchronous=True, frequency=frequency)
        # self.RC.control_interface.follow_trajectory({"positions":pos["right_arm"]}, "right_arm", asynchronous=True, frequency=frequency)
        # self.RC.control_interface.follow_trajectory({"positions":pos["left_arm"]}, "left_arm",  asynchronous=True, frequency=frequency)
        self.RC.sync()
    
    def control_gripper(self, left: Literal["none", "close", "open"], right: Literal["none", "close", "open"], overwrite_single_step=None):
        """
        Control both grippers. 
        """
        if self._fast_pass:
            return
        
        single_step = self._single_step if overwrite_single_step is None else overwrite_single_step
        if single_step:
            input(f"press enter to control grippers: {left} {right}\n")
        gripper_status = copy.deepcopy(getattr(self, "prev_gripper_status", dict(left=1., right=1.)))
        if left == "close":
            gripper_status["left"] = 0.
        elif left == "open":
            gripper_status["left"] = 1.
        if right == "close":
            gripper_status["right"] = 0.
        elif right == "open":
            gripper_status["right"] = 1.
        self.RC.set_grippers_status(gripper_status["left"], gripper_status["right"])
        self.prev_gripper_status = copy.deepcopy(gripper_status)
    
    def get_obs(self) -> Dict[Literal["depth", "clothes", "hanger", "rgb"], np.ndarray]:
        """
        Get observation. Use GroundSam to obtain the clothes' mask and the hanger's mask.

        Return:
        - depth: [H, W], float, in meter
        - clothes: [H, W], int, 0 or 1
        - hanger: [H, W], int, 0 or 1
        - rgb: [H, W, 3], uint8
        """
        if self._fast_pass:
            return dict(
                depth=np.ones((480, 640), dtype=np.float32) * 1.5,
                clothes=np.zeros((480, 640), dtype=np.int32),
                hanger=np.zeros((480, 640), dtype=np.int32),
                rgb=np.zeros((480, 640, 3), dtype=np.uint8),
            )
            
        obs = self.RC.get_rgbd()
        depth_processed, rgb = obs["depth"], obs["color"]
        
        img_rgb = Image.fromarray(rgb)
        imageio.imwrite(f"rgb_image.png", rgb)
        
        try:
            image_array, detections = self.GS.grounded_segmentation(img_rgb, ["clothes", "garment"])
            self.GS.plot_detections(image_array, detections, save_name=f"mask_clothes.png")
            mask_clothes = detections[0].mask
            plt.imshow(mask_clothes)
            plt.colorbar()
            plt.savefig(f'mask_clothes.png')
            plt.close()
        except IndexError as e:
            print(e)
            mask_clothes = np.zeros_like(depth_processed, dtype=np.int32)
        
        try:
            image_array, detections = self.GS.grounded_segmentation(img_rgb, "hanger")
            self.GS.plot_detections(image_array, detections, save_name=f"mask_hanger.png")
            mask_hanger = detections[0].mask
            plt.imshow(mask_hanger)
            plt.colorbar()
            plt.savefig(f'mask_hanger.png')
            plt.close()
        except IndexError as e:
            print(e)
            mask_hanger = np.zeros_like(depth_processed, dtype=np.int32)
        
        np.save("depth.npy", depth_processed)
        plt.imshow(depth_processed, cmap='gray')
        plt.savefig(f'depth.png')
        plt.colorbar()
        plt.close()
        
        return dict(
            depth=depth_processed,
            clothes=mask_clothes,
            hanger=mask_hanger,
            rgb=rgb,
        )
    
    def get_mask_use_sam(self, img: np.ndarray, input_point: np.ndarray, input_label: np.ndarray) -> np.ndarray:
        """
        Get mask using SAM model. 
        """
        masks, scores, logits = self.SAM.predict(img, input_point, input_label)
        return masks

    def get_head_camera_extrinsics(self):
        return self.RC.get_camera_extrinsics()
    
    def get_head_camera_intrinsics(self) -> np.ndarray:
        """
        Get head camera intrinsics.

        Return: 3*3 matrix
        """
        if self._fast_pass:
            return np.array([[100., 0., 320.], [0., 100., 240.], [0., 0., 1.]])
        return self.RC.get_camera_intrinsics()