import multiprocessing.managers
import cv2
from pdf2image import convert_from_path
import numpy as np
from typing import Optional, Tuple, Dict, Literal
import multiprocessing
import time
import open3d as o3d
import trimesh
import sys
import os
import signal
import threading
import taichi as ti
from numpy._typing import NDArray
import scipy.sparse.linalg as splinalg
from scipy.sparse import csc_matrix


from robohang.env.sapien_renderer import model_matrix_np, reproject
import robohang.sim.maths as maths


def _show_window_worker(img: np.ndarray, name: str, q):
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    cv2.imshow(name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    while q.empty():
        key = cv2.waitKey(100)
        if cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1.0: 
            break
    cv2.destroyWindow(name)


def _vis_pc_window_worker(xyz: np.ndarray, rgb: Optional[np.ndarray]):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    if rgb is not None:
        assert rgb.shape == xyz.shape, "{} != {}".format(rgb.shape, xyz.shape)
        pcd.colors = o3d.utility.Vector3dVector(rgb)
    o3d.visualization.draw_geometries([pcd])


class WindowVisualizer(object):
    def __init__(self) -> None:
        self._name_idx = 0
        self._enable = True
        
        self._q = multiprocessing.Manager().Queue()

    def enable(self, enable=True):
        self._enable = bool(enable)
    
    def stop_all(self):
        self._q.put("stop")
    
    def show(self, img: np.ndarray):
        if self._enable:
            self._name_idx += 1
            name = f"window_{self._name_idx}"
            p = multiprocessing.Process(target=_show_window_worker, args=(img, name, self._q), daemon=True)
            p.start()
    
    def vis_pc(self, xyz: np.ndarray, rgb: np.ndarray = None):
        if self._enable:
            p = multiprocessing.Process(target=_vis_pc_window_worker, args=(xyz, rgb), daemon=True)
            p.start()
        

vis = WindowVisualizer()


class SIGUSR1Exception(Exception):
    pass


try:
    import pynput

    class CtrlB_SIGUSR1_Catcher:
        def __init__(self):
            def handle_sigusr1(signum, frame):
                raise SIGUSR1Exception("Received SIGUSR1 signal!")
            signal.signal(signal.SIGUSR1, handle_sigusr1)
            print("register SIGUSR1 handler")

            self.ctrl_pressed = False
            self.listener_thread = threading.Thread(target=self.start_keyboard_listener, daemon=True)
            self.listener_thread.start()
            print("start CtrlB_SIGUSR1_Catcher listener_thread")

        def start_keyboard_listener(self):
            with pynput.keyboard.Listener(on_press=self.on_press, on_release=self.on_release) as listener:
                listener.join()
        
        def on_release(self, key):
            if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
                self.ctrl_pressed = False
        
        def on_press(self, key):
            if key == pynput.keyboard.Key.ctrl_l or key == pynput.keyboard.Key.ctrl_r:
                self.ctrl_pressed = True
            try:
                if key.char == "b" and self.ctrl_pressed:
                    print("Ctrl+B pressed, send SIGUSR1!")
                    os.kill(os.getpid(), signal.SIGUSR1)
            except AttributeError:
                pass
    
    catcher = CtrlB_SIGUSR1_Catcher()

except ImportError as e:
    print(f"{e}\nignore ...")

except Exception as e:
    raise e


def main():
    images = convert_from_path("outputs/2024-05-30/17-22-21/infer/insert_left/0/0.pdf")
    img = np.asarray(images[0])
    
    vis.vis_pc(trimesh.load("outputs/2024-06-01/15-22-25/full.ply").vertices)
    vis.vis_pc(trimesh.load("outputs/2024-06-01/15-22-25/full.ply").vertices, np.random.rand(*(trimesh.load("outputs/2024-06-01/15-22-25/full.ply").vertices.shape)))
    vis.show(img)
    time.sleep(5.)
    vis.show(img)
    
    for i in range(1000):
        time.sleep(1.)
        print(i)
        

if __name__ == "__main__":
    main()