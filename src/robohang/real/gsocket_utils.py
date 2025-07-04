import argparse
import time
import numpy as np
from typing import List, Dict, Literal
import cv2
import signal
from gsocket import BaseServer, BaseClient
from dataclasses import dataclass


IP, PORT = "192.168.50.127", 9018


@dataclass
class CameraExtrinsic:
    from_link: str
    to_link: str
    translation: List[float]
    rotation_xyzw: List[float]


class GalbotInterface:
    """Custom galbot charlie, use d436 head camera."""
    def __init__(self):
        import toml # type: ignore
        from galbot_control_interface import GalbotControlInterface # type: ignore
        from galbot_sensors.camera.realsense_camera import RealsenseCamera # type: ignore
        self.ci = GalbotControlInterface(sim=False, log_level="info")
        signal.signal(signal.SIGINT, signal.SIG_DFL)

        self.cam = RealsenseCamera({
            "serial_number": "213622076915",
            "camera_type": "Realsense D436",
            "camera_data_save_directory": "",
            "width": 640,
            "height": 480,
            "fps": 30,
            "color_format": "bgr8",
            "depth_format": "z16",
            "ir_format": "y8",
        })
        
        self.calib_result = toml.load("/usr/local/galbot/config/factory/calibration_config.toml")
    
    def hello_world(self):
        server_str = "This is Galbot Server, Hello World!"
        client_str = "Hello Client!"
        print(server_str)
        return client_str
    
    def get_rgbd(self) -> Dict[Literal["color", "depth"], np.ndarray]:
        f = self.cam.get_current_frames()
        color = cv2.cvtColor(f["color"], cv2.COLOR_BGR2RGB)
        depth = (f["depth"] / 1000.).astype(np.float16)
        return dict(color=color, depth=depth)
    
    def get_camera_intrinsics(self):
        return self.cam.get_intrinsics().get_intrinsics_matrix()
    
    def get_camera_extrinsics(self):
        return self.calib_result["front_camera"]

    def get_joints_name_list(self):
        return [
            "leg_joint1", "leg_joint2", "leg_joint3", "leg_joint4",
            "head_joint1", "head_joint2",
            "left_arm_joint1", "left_arm_joint2", "left_arm_joint3", "left_arm_joint4", 
            "left_arm_joint5", "left_arm_joint6", "left_arm_joint7",
            "right_arm_joint1", "right_arm_joint2", "right_arm_joint3", "right_arm_joint4", 
            "right_arm_joint5", "right_arm_joint6", "right_arm_joint7",
        ]
    
    def get_qpos(self):
        return np.array((
            self.ci.get_leg_joint_angles() + 
            self.ci.get_head_joint_angles() + 
            self.ci.get_arm_joint_angles(arm="left_arm") + 
            self.ci.get_arm_joint_angles(arm="right_arm")
        ), dtype=np.float32)
    
    def set_qpos(self, qpos: np.ndarray, speed=0.2, timeout=10., frequency=50., asynchronous=False):
        self.ci.set_leg_joint_angles(qpos[0:4].tolist(), speed=speed, asynchronous=True)
        self.ci.set_head_joint_angles(qpos[4:6].tolist(), speed=speed, asynchronous=True)
        self.ci.set_arm_joint_angles(qpos[6:13].tolist(), speed=speed, arm="left_arm", asynchronous=True)
        self.ci.set_arm_joint_angles(qpos[13:20].tolist(), speed=speed, arm="right_arm", asynchronous=True)
        while not asynchronous:
            stop_cnt = 0
            for hardware in ["right_arm", "left_arm", "head", "leg"]:
                traj_run, traj_stop_reason = self.ci.get_follow_trajectory_status(hardware)
                if not traj_run:
                    if 0 != traj_stop_reason:
                        return False
                    else:
                        stop_cnt += 1
            
            if stop_cnt == 4:
                break
            else:
                time.sleep(1 / frequency)
                timeout -= 1 / frequency
                if timeout < 0:
                    return False
        
        return True
    
    def sync(self, timeout=10., frequency=50.):
        while True:
            stop_cnt = 0
            for hardware in ["right_arm", "left_arm", "head", "leg"]:
                traj_run, traj_stop_reason = self.ci.get_follow_trajectory_status(hardware)
                if not traj_run:
                    if 0 != traj_stop_reason:
                        return False
                    else:
                        stop_cnt += 1
            for g in ["left_gripper", "right_gripper"]:
                if not self.ci.get_gripper_status(g)["running"]:
                    stop_cnt += 1
            
            if stop_cnt == 6:
                break
            else:
                time.sleep(1 / frequency)
                timeout -= 1 / frequency
                if timeout < 0:
                    return False
        return True
    
    def follow_trajectory_mul(self, hardwares, trajectorys_list, asynchronous):
        self.ci.follow_trajectory_mul(hardwares, trajectorys_list, asynchronous=asynchronous)
    
    def get_grippers_status(self):
        return dict(left=self.ci.get_gripper_status("left_gripper"), right=self.ci.get_gripper_status("right_gripper"))
    
    def set_grippers_action(self, left_gripper: float, right_gripper: float, speed=1., force=0.5, timeout=10., frequency=50., asynchronous=False):
        """0 is close, 1 is open"""
        self.ci.set_gripper_status(left_gripper, speed, force, "left_gripper", asynchronous=True)
        self.ci.set_gripper_status(right_gripper, speed, force, "right_gripper", asynchronous=True)
        while not asynchronous:
            stop_cnt = 0
            for g in ["left_gripper", "right_gripper"]:
                if not self.ci.get_gripper_status(g)["running"]:
                    stop_cnt += 1
                    
            if stop_cnt == 2:
                break
            else:
                time.sleep(1 / frequency)
                timeout -= 1 / frequency
                if timeout < 0:
                    return False
        
        return True


class GalbotClient(BaseClient):
    def __init__(self, host: str = IP, port: int = PORT):
        print("Start GalbotClient, if 'Hello Client!' is not printed, please try again.")
        super().__init__(host, port)
        print(self.hello_world())
    
    def hello_world(self) -> str:
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'hello_world', 'arguments': {}}
        return self._run(data)
    
    def get_rgbd(self) -> Dict[Literal["color", "depth"], np.ndarray]:
        """
        Returns:
        - color: np.ndarray, shape=(480, 640, 3), dtype=np.uint8
        - depth: np.ndarray, shape=(480, 640), dtype=np.float16
        """
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_rgbd', 'arguments': {}}
        return self._run(data)
    
    def get_camera_intrinsics(self) -> np.ndarray:
        """[3, 3] float64"""
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_camera_intrinsics', 'arguments': {}}
        return self._run(data)
    
    def get_camera_extrinsics(self) -> CameraExtrinsic:
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_camera_extrinsics', 'arguments': {}}
        ret = self._run(data)
        return CameraExtrinsic(
            from_link=ret["from_link"],
            to_link=ret["to_link"],
            translation=ret["translation"],
            rotation_xyzw=ret["rotation"],
        )
    
    def get_joints_name_list(self) -> List[str]:
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_joints_name_list', 'arguments': {}}
        return self._run(data)
    
    def get_qpos(self) -> np.ndarray:
        """[20] float32"""
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_qpos', 'arguments': {}}
        return self._run(data)
    
    def set_qpos(self, qpos: np.ndarray, speed=0.2, timeout=10., frequency=50., asynchronous=False) -> bool:
        """[20] float32"""
        qpos = np.array(qpos)
        assert qpos.shape == (20,)
        data = {
            'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'set_qpos', 
            'arguments': {'qpos': qpos, 'speed': speed, 
                          'timeout': timeout, 'frequency': frequency, 'asynchronous': asynchronous}
        }
        return self._run(data)
    
    def get_grippers_status(self):
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'get_grippers_status', 'arguments': {}}
        return self._run(data)
    
    def set_grippers_status(self, left_gripper: float, right_gripper: float, speed=1., force=0.5, timeout=10., frequency=50., asynchronous=False) -> bool:
        """0 is close, 1 is open"""
        data = {
            'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'set_grippers_action', 
            'arguments': {'left_gripper': left_gripper, 'right_gripper': right_gripper, 'speed': speed, 
                          'force': force, 'timeout': timeout, 'frequency': frequency, 'asynchronous': asynchronous}
        }
        return self._run(data)
    
    def sync(self):
        data = {'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'sync', 'arguments': {}}
        return self._run(data)
    
    def follow_trajectory_mul(self, hardwares: List[str], trajectorys_list: List[Dict[str, float]], asynchronous: bool):
        data = {
            'class': 'GalbotInterface', 'attribute_type': 'function', 'attribute_name': 'follow_trajectory_mul', 
            'arguments': {'hardwares': hardwares, 'trajectorys_list': trajectorys_list, 'asynchronous': asynchronous,}
        }
        self._run(data)


def demo_server():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default=IP, help="server ip address")
    parser.add_argument("--port", type=int, default=PORT, help="port number")
    args = parser.parse_args()
    
    robot = GalbotInterface()
    host = args.host
    port = args.port
    server = BaseServer(robot, host, port)


def demo_client():
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', type=str, default=IP, help='server ip address')
    parser.add_argument('--port', type=int, default=PORT, help='port number')
    args = parser.parse_args()
    
    host = args.host
    port = args.port
    client = GalbotClient(host, port)
    
    rgbd = client.get_rgbd()
    color, depth = rgbd['color'], rgbd['depth']
    print(color.shape, color.dtype)
    print(depth.shape, depth.dtype)
    
    camera_intrinsics = client.get_camera_intrinsics()
    print(camera_intrinsics.shape, camera_intrinsics.dtype)
    
    camera_extrinsics = client.get_camera_extrinsics()
    print(camera_extrinsics)
    
    joints_name_list = client.get_joints_name_list()
    print(joints_name_list)
    
    qpos = client.get_qpos()
    print(qpos.shape, qpos.dtype)
    print(qpos)
    
    qpos[6] += 0.01
    client.set_qpos(qpos)
    
    print(client.get_grippers_status())
    client.set_grippers_status(0., 0.)
    client.set_grippers_status(1., 1.)
    

def test_server():
    robot = GalbotInterface()
    # qpos = robot.get_qpos()
    # qpos[6] += 0.01
    # robot.set_qpos(qpos)
    
    # print(robot.get_qpos())
    # robot.ci.set_gripper_close()
    # robot.set_grippers_action(0., 0.)
    # robot.set_grippers_action(1., 1.)


if __name__ == "__main__":
    # test_server()
    demo_server()
