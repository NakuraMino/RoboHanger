import pyglet
pyglet.options['shadow_window'] = False
import os
os.environ['PYOPENGL_PLATFORM'] = 'egl'
import numpy as np
import trimesh
import trimesh.transformations as tra

import sapien.core as sapien
import open3d as o3d
from sklearn.decomposition import PCA

import omegaconf
from typing import Dict, Literal

class SapienRenderer():
    def __init__(self, device="cuda") -> None:
        self._engine = sapien.Engine()
        self._renderer = sapien.SapienRenderer(offscreen_only=True, device=device)
        self._engine.set_renderer(self._renderer)
        sapien.render_config.camera_shader_dir = "ibl"
        
        scene_config = sapien.SceneConfig()
        scene_config.gravity = np.array([0., 0., 0.])
        self._scene = self._engine.create_scene(scene_config)

        self._scene.add_point_light([+2., +2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([-2., +2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([+2., -2., 4.], [20.0, 20.0, 20.0])
        self._scene.add_point_light([-2., -2., 4.], [20.0, 20.0, 20.0])

        self._hanger = None
    
    def set_scene(self, hanger_path: str, hanger_pose: sapien.Pose):
        need_clear = True
        if self._hanger is not None:
            self._scene.remove_actor(self._hanger)
            need_clear = True
        
        if need_clear:
            self._renderer.clear_cached_resources()

        self._camera = self._scene.add_camera(
            name="camera", 
            width=320, height=320, 
            fovy=np.deg2rad(60), near=0.1, far=10.
        )
        
        mask_id = {}

        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [1., 1., 1., 1.]
        material.specular = .5
        material.roughness = .5
        material.metallic = .0
        builder.add_visual_from_file(filename=hanger_path, material=material)
        self._hanger = builder.build_static(name="hanger")
        self._hanger.set_pose(hanger_pose)
        
        mask_id["hanger"] = self._hanger.get_id()

        return mask_id

    @property
    def camera(self):
        return self._camera
    
    def render(self) -> Dict[Literal["depth", "rgba", "mask", "pcd"], np.ndarray]:
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

        points_opengl = position[..., :3][position[..., 3] < 1]
        model_matrix = camera.get_model_matrix()
        points_world = points_opengl @ model_matrix[:3, :3].T + model_matrix[:3, 3]
        return_val["pcd"] = points_world

        self._scene.remove_camera(self._camera)
        return return_val


def render_pcd(
        hanger_obj_path : str,
        hanger_pose : np.ndarray,
        sapien_renderer : SapienRenderer,
) -> np.ndarray:
    
    """ render point_cloud for a specific hanger in hanger_pose """

    _, _, angle, translate, _ = tra.decompose_matrix(hanger_pose)
    sapien_renderer.set_scene(
        hanger_obj_path,
        sapien.Pose(p=translate, q=tra.quaternion_from_euler(*angle))
    )
    render_result = sapien_renderer.render()
    point_cloud = render_result["pcd"]

    return point_cloud


def get_label_point(
        hanger_obj_path : str,
        transformation : np.ndarray,
) -> np.ndarray:
    
    """ generate label points """

    label_dir = os.path.split(hanger_obj_path)[0]
    label_path = os.path.join(label_dir, "hanger.obj.meta.yaml")
    
    labels = omegaconf.OmegaConf.load(label_path)

    label_raw = np.zeros((2, 3))
    label_raw[0, :] = labels["left"]
    label_raw[1, :] = labels["right"]
    label_fine = np.zeros((3, 3))
    label_fine[2, :] = transformation[:3, 3]

    ones = np.ones((label_raw.shape[0], 1))
    label_raw = np.concatenate([label_raw, ones], axis=1).T
    label_raw = np.matmul(transformation, label_raw)[:3, :].T
    label_fine[:2, :] = label_raw

    return label_fine


def data_post_process(
        point_cloud : np.ndarray,
        label_point : np.ndarray,
        sample_num : int,
        target_dir : str, 
):

    """ PCA + sample + randomize + move_to_origin + save_to_target_dir """

    ### PCA
    data = np.concatenate([point_cloud, label_point], axis=0)
    pca = PCA(n_components=3)
    _ = pca.fit_transform(data)
    n_components = pca.components_

    transform_mat = np.zeros((3,3), dtype=np.float32)
    transform_mat[:2, :] = n_components[:2, :]
    new_verts = (data-data.mean(axis=0)) @ transform_mat.T @ n_components # [N+3, 3]

    pcd_projected = new_verts[:-3, :]
    lbl_projected = new_verts[-3:, :]

    ### sample
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pcd_projected)
    pcd_sampled = np.array(pcd.farthest_point_down_sample(sample_num).points)

    ### randomize
    pcd_randomized = pcd_sampled + np.random.normal(loc=0, scale=0.003, size=pcd_sampled.shape)

    ### move_to_origin
    translation = pcd_randomized.mean(axis=0)
    pcd_centralized = pcd_randomized - translation
    lbl_centralized = lbl_projected - translation

    ### save_to_target_dir
    trimesh.PointCloud(vertices=pcd_centralized).export(os.path.join(target_dir, "pcd.ply"))
    np.save(os.path.join(target_dir, "label.npy"), lbl_centralized)


def generate_random_pcd(
        hanger_obj_path : str,
        sapien_renderer : SapienRenderer, 
        num_to_generate : np.int32 = 1,
        num_sample_point : np.int32 = 512,
        target_dir : str = "outputs"
):
    
    """ 
        For one given obj, randomize its pose and generate
        corresponding point clouds, save them in 'target_dir'. 
    """

    def rand(left_bound, right_bound):

        """ generate random float in [left_bound, right_bound) """

        return left_bound + np.random.rand() * (right_bound - left_bound)

    ## define limits
    rotation_limit = np.array([[-np.pi*2, +np.pi*2],
                               [-np.pi/6, +np.pi/6], 
                               [-np.pi/6, +np.pi/6]])
    
    translation_limit = np.array([[-0.05, +0.05], 
                                  [-0.05, +0.05], 
                                  [-0.05, +0.05]])

    for i in range(num_to_generate):
        r_x = rand(rotation_limit[0, 0], rotation_limit[0, 1])
        r_y = rand(rotation_limit[1, 0], rotation_limit[1, 1])
        r_z = rand(rotation_limit[2, 0], rotation_limit[2, 1])
        t_x = rand(translation_limit[0, 0], translation_limit[0, 1])
        t_y = rand(translation_limit[1, 0], translation_limit[1, 1])
        t_z = rand(translation_limit[2, 0], translation_limit[2, 1])


        ## (0.7, 0, 0) is set as the the base pose, so (t_x + 0.7) is used here
        base_rotation = tra.euler_matrix(0, np.pi/2, 0)
        rotation = np.matmul(tra.euler_matrix(r_x, r_y, r_z), base_rotation)
        M = tra.compose_matrix(angles=tra.euler_from_matrix(rotation), translate=(t_x+0.7, t_y, t_z))

        hanger_pose = M

        ## save point cloud and label points
        dir = os.path.join(target_dir, f"{i}")
        os.makedirs(dir, exist_ok=True)

        point_cloud = render_pcd(hanger_obj_path, hanger_pose, sapien_renderer)
        label_point = get_label_point(hanger_obj_path, hanger_pose)

        data_post_process(point_cloud, label_point, num_sample_point, dir)    

