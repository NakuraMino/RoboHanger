from typing import List, Dict, Optional, Literal

from dataclasses import dataclass

import numpy as np
import trimesh.transformations as tra

import omegaconf

import sapien.core as sapien
from sapien.utils.viewer import Viewer


@dataclass
class HangerVisualizerMode:
    ee_link: str
    origin: np.ndarray


class SapienVisualize:
    def __init__(
        self,
        viewer_cfg: omegaconf.DictConfig,
        urdf_path: str,
        hanger_path: str,
        table_height: float, 
        device="cuda:0",
    ) -> None:
        self._engine = sapien.Engine()
        sapien.render_config.viewer_shader_dir = "rt"
        sapien.render_config.rt_samples_per_pixel = int(viewer_cfg.rt_samples_per_pixel)
        sapien.render_config.rt_use_denoiser = bool(viewer_cfg.rt_use_denoiser)
        self._renderer = sapien.SapienRenderer(device=device)
        self._engine.set_renderer(self._renderer)

        scene_config = sapien.SceneConfig()
        self._timestep = float(viewer_cfg.timestep)
        self._scene = self._engine.create_scene(scene_config)
        self._scene.set_timestep(self._timestep)

        # ground
        material = self._renderer.create_material()
        material.base_color = [0.0, 0.0, 0.0, 1.]
        self._scene.add_ground(0, render_material=material)

        # table
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [0., 0.5, 0., 1.]
        builder.add_box_visual(half_size=[1.0, 0.5, table_height / 2], material=material)
        self._table = builder.build_static(name="table")
        self._table.set_pose(pose=sapien.Pose(p=[0., 0.5, table_height / 2]))

        # hanger
        builder = self._scene.create_actor_builder()
        material = self._renderer.create_material()
        material.base_color = [0.5, 0., 0., 1.]
        builder.add_visual_from_file(hanger_path, material=material)
        self._hanger = builder.build_static(name="hanger")
        self._hanger.set_pose(pose=sapien.Pose.from_transformation_matrix(np.array(
            [[0.98085529, 0.19473806, 0., 0.39290115],
             [-0.19473806,0.98085529, 0., 0.63531607],
             [0., 0., 1., 0.47],
             [0., 0., 0., 1.]]
        )))

        # light
        self._scene.set_ambient_light([0.5, 0.5, 0.5])
        self._scene.add_directional_light([0, 1, -1], [0.5, 0.5, 0.5])

        # viewer
        self._viewer = Viewer(self._renderer, resolutions=viewer_cfg.resolution)
        self._viewer.set_scene(self._scene)
        # self._viewer.set_camera_xyz(x=-1.5, y=+1.5, z=1.5)
        # self._viewer.set_camera_rpy(r=0, p=-0.4, y=np.pi / 4)
        self._viewer.set_camera_xyz(x=0., y=+1.5, z=2.0)
        self._viewer.set_camera_rpy(r=0., p=-0.9, y=np.pi / 2)
        # self._viewer.set_camera_xyz(x=-3.0, y=0.0, z=0.5)
        # self._viewer.set_camera_rpy(r=0., p=0., y=0.)
        self._viewer.set_fovy(1.0)

        # Load URDF
        loader: sapien.URDFLoader = self._scene.create_urdf_loader()
        loader.fix_root_link = True
        self._robot = loader.load_kinematic(urdf_path)
        self._sapien_notfixed_jnames: List[str] = []
        for j in self._robot.get_joints():
            if j.type != "fixed":
                self._sapien_notfixed_jnames.append(j.name)
        self._robot.set_root_pose(sapien.Pose([0, 0, 0], [1, 0, 0, 0]))

    @property
    def robot(self):
        return self._robot
    
    @property
    def closed(self):
        return self._viewer.closed
    
    @property
    def sapien_notfixed_jnames(self):
        return self._sapien_notfixed_jnames
    
    @property
    def timestep(self):
        return self._timestep

    def step(
        self, 
        cfg: Optional[Dict[str, float]]=None, 
        root_mat: Optional[np.ndarray]=None, 
        hanger_mode: Optional[HangerVisualizerMode]=None,
    ):
        if root_mat is not None:
            self._robot.set_root_pose(sapien.Pose(p=root_mat[:3, 3], q=tra.quaternion_from_matrix(root_mat)))
        if cfg is not None:
            self._robot.set_qpos(np.array([cfg[k] for k in self._sapien_notfixed_jnames]))
        self._scene.step()
        if hanger_mode is not None:
            for l in self._robot.get_links():
                if l.name == hanger_mode.ee_link:
                    mat = l.get_pose().to_transformation_matrix()
                    self._hanger.set_pose(sapien.Pose.from_transformation_matrix(mat @ hanger_mode.origin))
                    break
                '''if l.name == "torso_base_link":
                    print(l.get_pose())'''
        self._scene.step()

    def render(self):
        self._scene.update_render()
        self._viewer.render()


def test_sapien():
    vis = SapienVisualize(
        omegaconf.DictConfig(dict(rt_samples_per_pixel=4, rt_use_denoiser=True, timestep=0.01, resolution=[1440, 1440])),
        urdf_path="assets/robot/galbot_one_charlie/urdf.urdf",
        hanger_path="assets/hanger/0/hanger_vis.obj",
        table_height=0.455,
    )
    vis.render()


if __name__ == "__main__":
    test_sapien()