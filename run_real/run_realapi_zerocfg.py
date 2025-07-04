from robohang.real.realapi import RealAPI
import json
import omegaconf
import sys, os
import time as time_module
import matplotlib.pyplot as plt
import cv2
import numpy as np

realapi = RealAPI(omegaconf.DictConfig(dict(single_step=True, fast_pass=False)))
print("test gripper ...")
realapi.control_gripper("open", "open")
realapi.control_gripper("close", "close")
realapi.control_gripper("open", "open")
print("zero cfg")

# left_arms = np.array([0.5, 0.5, -2.5, -1.5, 1.0, 1.0, -0.5])
# realapi.move_to_qpos(
#     dict(
#         leg_joint1=0.3, leg_joint2=0.6, leg_joint3=0.3, leg_joint4=0.0,
#         left_arm_joint1=+0.0,  left_arm_joint2=+0.4,  left_arm_joint3=-1.5,  left_arm_joint4=-1.5,  left_arm_joint5=0.0,  left_arm_joint6=+0.5,  left_arm_joint7=-0.5,
#         right_arm_joint1=-0.0, right_arm_joint2=-0.4, right_arm_joint3=+1.5, right_arm_joint4=+1.5, right_arm_joint5=0.0, right_arm_joint6=-0.5, right_arm_joint7=+0.5,
#     )
# )

realapi.move_to_qpos(
    dict(
        leg_joint1=0.3, leg_joint2=0.6, leg_joint3=0.3, leg_joint4=0.0,
        left_arm_joint1=0.0, left_arm_joint2=0.0, left_arm_joint3=0.0,
        left_arm_joint4=0.0, left_arm_joint5=0.0, left_arm_joint6=0.0, left_arm_joint7=0.0,
        right_arm_joint1=0.0, right_arm_joint2=0.0, right_arm_joint3=0.0,
        right_arm_joint4=0.0, right_arm_joint5=0.0, right_arm_joint6=0.0, right_arm_joint7=0.0,
    )
)

