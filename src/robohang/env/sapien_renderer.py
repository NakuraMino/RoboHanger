import logging
logger = logging.getLogger(__name__)

import taichi as ti

from typing import Optional, Dict, Literal, Tuple, List
import colorsys
import json
import os
import copy
from dataclasses import dataclass

import numpy as np
from numpy._typing import NDArray
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix

from PIL import ImageColor, Image
import matplotlib.pyplot as plt
import tqdm
import trimesh.transformations as tra
import omegaconf

import sapien.core as sapien

import robohang.sim.maths as maths


@dataclass
class CameraProperty:
    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float
    skew: float

    def to_dict(self):
        return {attr: getattr(self, attr) for attr in self.__dict__.keys()}
    

model_matrix_np = np.array([
    [ 0.,  0.,  1.,  0.],
    [-1.,  0.,  0.,  0.],
    [ 0., -1.,  0.,  0.],
    [ 0.,  0.,  0.,  1.],
]) # np.linalg.inv(tra.euler_matrix(np.pi / 2, -np.pi / 2, 0))


def randomize_camera(
    camera_prop: CameraProperty, 
    camera_pose: List[float],
    rand_cfg: omegaconf.DictConfig,
):
    camera_prop: CameraProperty = copy.deepcopy(camera_prop)
    camera_pose: List[float] = copy.deepcopy(camera_pose)

    camera_prop.cx += (np.random.random() * 2. - 1.) * rand_cfg.prop.cx
    camera_prop.cy += (np.random.random() * 2. - 1.) * rand_cfg.prop.cy
    camera_prop.fx += (np.random.random() * 2. - 1.) * rand_cfg.prop.fx
    camera_prop.fy += (np.random.random() * 2. - 1.) * rand_cfg.prop.fy
    camera_pose[:3] = (np.array(camera_pose[:3]) + (np.random.random(3) * 2. - 1.) * rand_cfg.pose.xyz).tolist()
    camera_pose[3:] = tra.quaternion_from_matrix(
        tra.rotation_matrix(rand_cfg.pose.rot, np.random.randn(3)) @
        tra.quaternion_matrix(camera_pose[3:])
    ).tolist()
    return camera_prop, camera_pose


def camera_property_to_intrinsics_matrix(prop: CameraProperty, dtype=np.float32):
    return np.array([
        [prop.fx, prop.skew, prop.cx],
        [0., prop.fy, prop.cy],
        [0., 0., 1.]
    ], dtype=dtype)


def camera_pose_to_matrix(pos7d: list, dtype=np.float32):
    return (tra.translation_matrix(pos7d[:3]) @ tra.quaternion_matrix(pos7d[3:])).astype(dtype=dtype)


def xyz2uv(xyz: np.ndarray, extrinsic: np.ndarray, intrinsic: np.ndarray):
    assert xyz.shape == (3, ), xyz.shape
    assert extrinsic.shape == (4, 4), extrinsic.shape
    assert intrinsic.shape == (3, 3), intrinsic.shape
    xyzw_w = np.array([*xyz, 1.], dtype=np.float32)
    xyzw_c = np.linalg.inv(extrinsic @ model_matrix_np) @ xyzw_w
    xyz_c = xyzw_c[:3] / xyzw_c[3]
    uvw = intrinsic @ (xyz_c / xyz_c[2])
    uvw /= uvw[2]
    return uvw[:2]


@ti.kernel
def _reproject_kernel(
    depth_input: ti.types.ndarray(dtype=float),
    mask_input: ti.types.ndarray(dtype=int),

    depth_output: ti.types.ndarray(dtype=float),
    mask_output: ti.types.ndarray(dtype=int),
    output_is_hole: ti.types.ndarray(dtype=int),

    intrinsics_matrix_input: ti.math.mat3,
    intrinsics_matrix_output: ti.math.mat3,
    camera_pose_input: ti.math.mat4,
    camera_pose_output: ti.math.mat4,

    x1x2y1y2: ti.math.vec4, 
    far_output: float,
):
    model_matrix = ti.Matrix(model_matrix_np, dt=float)
    for i, j in ti.ndrange(depth_output.shape[0], depth_output.shape[1]):
        depth_output[i, j] = far_output
        mask_output[i, j] = 0
        output_is_hole[i, j] = 1
    
    # reproject
    x1, x2, y1, y2 = x1x2y1y2
    for i, j in ti.ndrange(depth_input.shape[0], depth_input.shape[1]):
        xyz_i = intrinsics_matrix_input.inverse() @ ti.Vector([j + 0.5, i + 0.5, 1.], dt=float)
        xyz_i *= (depth_input[i, j] / xyz_i[2])
        xyzw_i = maths.vec3_pad1_func(xyz_i)
        xyzw_w = camera_pose_input @ model_matrix @ xyzw_i
        xyzw_o = model_matrix.inverse() @ camera_pose_output.inverse() @ xyzw_w
        xyz_o = xyzw_o[:3] / xyzw_o[3]
        if xyz_o[2] > 0. and x1 < xyzw_w[0] and xyzw_w[0] < x2 and y1 < xyzw_w[1] and xyzw_w[1] < y2:
            d = xyz_o[2]
            uvw = intrinsics_matrix_output @ (xyz_o / d)
            uvw /= uvw[2]
            u, v, _ = uvw
            i_output = ti.floor(v, dtype=int)
            j_output = ti.floor(u, dtype=int)
            if (
                (0 <= i_output and i_output < depth_output.shape[0]) and
                (0 <= j_output and j_output < depth_output.shape[1])
            ):
                ti.atomic_min(depth_output[i_output, j_output], d)
                ti.atomic_or(mask_output[i_output, j_output], mask_input[i, j])
                output_is_hole[i_output, j_output] = 0


@ti.kernel
def _make_matrix_kernel(
    output_is_hole: ti.types.ndarray(dtype=int),
    row: ti.types.ndarray(dtype=int),
    col: ti.types.ndarray(dtype=int),
    val: ti.types.ndarray(dtype=float),

    depth_output: ti.types.ndarray(dtype=float),
    depth_rhs: ti.types.ndarray(dtype=float),

    mask_output: ti.types.ndarray(dtype=float),
    mask_rhs: ti.types.ndarray(dtype=float), 
) -> int:
    triplet_cnt = 0
    u = ti.Vector([0, +1, 0, -1, 0], dt=int)
    v = ti.Vector([0, 0, +1, 0, -1], dt=int)

    for i, j in ti.ndrange(depth_output.shape[0], depth_output.shape[1]):
        ij = i * depth_output.shape[1] + j
        if output_is_hole[i, j] == 1:
            depth_rhs[ij] = mask_rhs[ij] = 0.

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
            mask_rhs[ij] = mask_output[i, j]

            tid = ti.atomic_add(triplet_cnt, +1)
            col[tid] = ij
            row[tid] = ij
            val[tid] = 1.
                        
    return triplet_cnt


def reproject(
    depth_input: NDArray[np.float32],
    mask_input: NDArray[np.int32],
    output_shape: Tuple[int, int], 
    intrinsics_matrix_input: np.ndarray,
    intrinsics_matrix_output: np.ndarray,
    camera_pose_input: np.ndarray,
    camera_pose_output: np.ndarray,
    interp_mask: bool,
    x1x2y1y2: np.ndarray,
    far_output=10.,
    dtype=np.float32,
    dtype_int=np.int32,
) -> Dict[Literal["depth_output", "mask_output", "output_is_hole"], np.ndarray]:
    depth_output = np.zeros(output_shape, dtype=dtype)
    mask_int_output = np.zeros(output_shape, dtype=dtype_int)
    output_is_hole = np.zeros_like(depth_output, dtype=dtype_int)
    _reproject_kernel(
        depth_input,
        mask_input,
        depth_output,
        mask_int_output,
        output_is_hole,
        intrinsics_matrix_input,
        intrinsics_matrix_output,
        camera_pose_input,
        camera_pose_output,
        x1x2y1y2,
        far_output,
    )

    mask_output = mask_int_output.astype(dtype)
    depth_rhs = np.zeros_like(depth_output, dtype=dtype).flatten()
    mask_rhs = np.zeros_like(depth_output, dtype=dtype).flatten()
    row = np.zeros(depth_rhs.shape[0] * 5, dtype=dtype_int)
    col = np.zeros(depth_rhs.shape[0] * 5, dtype=dtype_int)
    val = np.zeros(depth_rhs.shape[0] * 5, dtype=dtype)
    
    triplet_cnt = _make_matrix_kernel(
        output_is_hole,
        row,
        col,
        val,
        depth_output,
        depth_rhs,
        mask_output,
        mask_rhs,
    )

    mat = csc_matrix((val[:triplet_cnt], (row[:triplet_cnt], col[:triplet_cnt])), shape=(depth_rhs.shape[0], depth_rhs.shape[0]))
    return dict(
        depth_output=splinalg.spsolve(mat, depth_rhs).reshape(depth_output.shape),
        mask_output=(
            splinalg.spsolve(mat, mask_rhs).reshape(mask_output.shape) if interp_mask else
            mask_rhs.reshape(mask_output.shape)
        ),
        output_is_hole=output_is_hole,
    )


class SapienRenderer:
    def __init__(
        self,
        robot_urdf: str,
        table_path: str,
        hanger_path: str,
        device="cuda",
    ):
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer(offscreen_only=True, device=device)
        self._engine.set_renderer(self._renderer)
        sapien.render_config.camera_shader_dir = "ibl"

        scene_config = sapien.SceneConfig()
        scene_config.gravity = np.array([0., 0., 0.])
        self._scene = self._engine.create_scene(scene_config)

        logger.info("load urdf ...")
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load_kinematic(robot_urdf)
        self._sapien_notfixed_jnames: List[str] = []
        for j in self._robot.get_joints():
            if j.type != "fixed":
                self._sapien_notfixed_jnames.append(j.name)
        logger.info("load urdf completed ...")

        self._scene.add_point_light([+2., +2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([-2., +2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([+2., -2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([-2., -2., 4.], [20.0, 20.0, 20.0])

        self._garment = None
        self._garment_inv = None
        # self._table = None
        # self._hanger = None
        self._camera = None
        self._gripper_l = None
        self._gripper_r = None

        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [.2, .2, .2, 1.]
        material.specular = .0
        material.roughness = .5
        material.metallic = .0
        builder.add_visual_from_file(filename=table_path, material=material)
        self._table = builder.build_static(name="table")

        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [1., 1., 1., 1.]
        material.specular = .5
        material.roughness = .5
        material.metallic = .0
        builder.add_visual_from_file(filename=hanger_path, material=material)
        self._hanger = builder.build_static(name="hanger")

    @property
    def scene(self):
        return self._scene

    def set_scene(
        self,
        camera_prop: CameraProperty,
        camera_pose: sapien.Pose,
        garment_filename: Optional[str]=None,
        garment_inv_filename: Optional[str]=None,
        table_filename: Optional[str]=None,
        hanger_filename: Optional[str]=None,
        robot_filename: Optional[str]=None,
        gripper_l_filename: Optional[str]=None,
        gripper_r_filename: Optional[str]=None,
        garment_hsv=(0.8444218515250481, 0.7579544029403025, 0.7102857904154225)
    ) -> Dict[str, int]:
        """return mask_id: Map [str -> int]"""
        mask_id = {}

        need_clear = False
        if self._garment is not None:
            self._scene.remove_actor(self._garment)
            need_clear = True
        if self._garment_inv is not None:
            self._scene.remove_actor(self._garment_inv)
            need_clear = True
        if self._table is not None and False:
            self._scene.remove_actor(self._table)
            need_clear = True
        if self._hanger is not None and False:
            self._scene.remove_actor(self._hanger)
            need_clear = True
        if self._gripper_l is not None:
            self._scene.remove_actor(self._gripper_l)
            need_clear = True
        if self._gripper_r is not None:
            self._scene.remove_actor(self._gripper_r)
            need_clear = True
        if self._camera is not None:
            self._scene.remove_camera(self._camera)
            need_clear = True
        if need_clear:
            self._renderer.clear_cached_resources()

        self._camera = self._scene.add_camera(
            name="camera", 
            width=camera_prop.width, height=camera_prop.height, 
            fovy=np.deg2rad(35), near=0.2, far=10.
        )
        self._camera.set_perspective_parameters(
            fx=camera_prop.fx, fy=camera_prop.fy, 
            cx=camera_prop.cx, cy=camera_prop.cy, 
            skew=camera_prop.skew, near=0.2, far=10.
        )
        self._camera.set_pose(camera_pose)
        # logger.info(f"set_pose:{camera_pose}")
            
        if garment_filename is not None:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [*colorsys.hsv_to_rgb(*garment_hsv), 1.]
            material.specular = .5
            material.roughness = .8
            material.metallic = .0
            builder.add_visual_from_file(filename=garment_filename, material=material)
            self._garment = builder.build_static(name="garment")
            mask_id["garment"] = self._garment.get_id()

        if garment_inv_filename is not None:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [*colorsys.hsv_to_rgb(*garment_hsv), 1.]
            material.specular = .5
            material.roughness = .8
            material.metallic = .0
            builder.add_visual_from_file(filename=garment_inv_filename, material=material)
            self._garment_inv = builder.build_static(name="garment_inv")
            mask_id["garment_inv"] = self._garment_inv.get_id()

        if table_filename is not None and False:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [.2, .2, .2, 1.]
            material.specular = .0
            material.roughness = .5
            material.metallic = .0
            builder.add_visual_from_file(filename=table_filename, material=material)
            self._table = builder.build_static(name="table")
            mask_id["table"] = self._table.get_id()
        
        if hanger_filename is not None and False:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [1., 1., 1., 1.]
            material.specular = .5
            material.roughness = .5
            material.metallic = .0
            builder.add_visual_from_file(filename=hanger_filename, material=material)
            self._hanger = builder.build_static(name="hanger")
            mask_id["hanger"] = self._hanger.get_id()
        
        if table_filename is not None:
            self._table.set_pose(sapien.Pose.from_transformation_matrix(np.load(table_filename)))
            mask_id["table"] = self._table.get_id()
        
        if hanger_filename is not None:
            self._hanger.set_pose(sapien.Pose.from_transformation_matrix(np.load(hanger_filename)))
            mask_id["hanger"] = self._hanger.get_id()
        
        if gripper_l_filename is not None:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [1., 1., 1., 1.]
            material.specular = .0
            material.roughness = .5
            material.metallic = .0
            builder.add_visual_from_file(filename=gripper_l_filename, material=material)
            self._gripper_l = builder.build_static(name="gripper_l")
            mask_id["gripper_l"] = self._gripper_l.get_id()
        
        if gripper_r_filename is not None:
            builder = self._scene.create_actor_builder()
            material = self._renderer.create_material()
            material.base_color = [1., 1., 1., 1.]
            material.specular = .0
            material.roughness = .5
            material.metallic = .0
            builder.add_visual_from_file(filename=gripper_r_filename, material=material)
            self._gripper_r = builder.build_static(name="gripper_r")
            mask_id["gripper_r"] = self._gripper_r.get_id()

        if robot_filename is not None:
            with open(robot_filename, "r") as f_obj:
                robot_cfg = json.load(f_obj)

            root_mat = np.array(robot_cfg["base_mat"])
            self._robot.set_root_pose(sapien.Pose(root_mat[:3, 3], tra.quaternion_from_matrix(root_mat)))

            self._robot.set_qpos(np.array([robot_cfg["pos_cfg"][k] for k in self._sapien_notfixed_jnames]))
        
        return mask_id
    
    def render(self) -> Dict[Literal["depth", "rgba", "mask"], np.ndarray]:
        return_val = {}

        self._scene.step()
        self._scene.update_render()

        camera = self._camera
        camera.take_picture()

        position = camera.get_float_texture("Position") # [H, W, 4]
        far, near = camera.far, camera.near
        return_val["depth"] = far * near / (far - (far - near) * position[..., 3])

        rgba = camera.get_float_texture("Color") # [H, W, 4]
        rgba = (rgba * 255).clip(0, 255).astype("uint8")
        return_val["rgba"] = rgba

        seg_labels = camera.get_uint32_texture('Segmentation')  # [H, W, 4]
        label1_image = seg_labels[..., 1].astype(np.int32)  # actor-level
        return_val["mask"] = label1_image

        return return_val


def main():
    test_path = "/mnt/disk1/yuxingchen/research/robohang/outputs/test/2024-03-04/20-14-34"
    sapien_renderer = SapienRenderer(
        robot_urdf="/mnt/disk1/yuxingchen/research/robohang/assets/robot/galbot_zero_description/galbot_zero_two_grippers.urdf"
    )
    sapien_renderer.set_camera_pos(
        xyz=np.array([-4., -4., 6.]), 
        quat=tra.quaternion_from_matrix(tra.euler_matrix(0., np.pi/4, np.pi/4))
    )

    N = 80
    result_list = []
    for i in tqdm.tqdm(range(N)):
        base_dir = f"{test_path}/obj/{str(i).zfill(6)}"
        sapien_renderer.set_scene(
            garment_filename=os.path.join(base_dir, "garment.obj"),
            table_filename=os.path.join(base_dir, "table.obj"),
            robot_filename=os.path.join(base_dir, "robot.json"),
        )
        result_list.append(sapien_renderer.render())

    output_dir = f"{test_path}/sapien/"
    os.makedirs(os.path.join(output_dir, "depth"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "color"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "mask"), exist_ok=True)
    
    colormap = sorted(set(ImageColor.colormap.values()))
    color_palette = np.array([ImageColor.getrgb(color) for color in colormap], dtype=np.uint8)
    for i in tqdm.tqdm(range(N)):
        result = result_list[i]
        plt.figure()
        plt.imshow(np.clip(result["depth"], 0., 5.))
        plt.colorbar()
        plt.savefig(os.path.join(output_dir, "depth", f"{str(i).zfill(6)}.png"))
        plt.close()
        plt.imsave(os.path.join(output_dir, "color", f"{str(i).zfill(6)}.png"), result["rgba"])

        label1_pil = Image.fromarray(color_palette[result["mask"] % len(color_palette)])
        label1_pil.save(os.path.join(output_dir, "mask", f"{str(i).zfill(6)}.png"))
    

if __name__ == "__main__":
    main()