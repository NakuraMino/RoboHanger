from robohang.real.realapi import RealAPI
import json
import omegaconf
import sys, os
import time as time_module
import matplotlib.pyplot as plt
import cv2
import trimesh.transformations as tra
import numpy as np

realapi = RealAPI(omegaconf.DictConfig(dict(single_step=True, fast_pass=False)))
if True:
    name = "rgb"
    cv2.namedWindow(name, cv2.WINDOW_NORMAL)
    while True:
        key = cv2.waitKey(1)
        rgbd = realapi.RC.get_rgbd()
        rgb = rgbd["color"]
        print(realapi.get_head_camera_extrinsics())
        print(realapi.get_head_camera_intrinsics())
        print(tra.euler_from_matrix(
            tra.translation_matrix(realapi.get_head_camera_extrinsics().translation) @ 
            tra.quaternion_matrix(np.array(realapi.get_head_camera_extrinsics().rotation_xyzw)[[1, 2, 3, 0]])
        ))
        cv2.imshow(name, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        if (cv2.getWindowProperty(name, cv2.WND_PROP_VISIBLE) < 1.0) or (key & 0xFF == ord('q')): 
            break
    cv2.destroyWindow(name)