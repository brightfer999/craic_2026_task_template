#!/usr/bin/env python3

import rospy
import math
from sensor_msgs.msg import JointState
from geometry_msgs.msg import Point
from typing import List, Optional


DEFAULT_ARM_POSES = {
    "home": [0.0] * 14,
    "forward": [0.5, 0.0, 0.0, -1.57, 0.0, 0.0, 0.0] * 2,
    "grasp_left": [
        0.5, 0.0, -0.3, -2.0, 0.0, 0.0, 0.5,
    ] + [0.0] * 7,
    "grasp_right": [0.0] * 7 + [
        0.5, 0.0, -0.3, -2.0, 0.0, 0.0, 0.5,
    ],
    "up": [0.0, 0.0, 0.0, -0.5, 0.0, 0.0, 0.0] * 2,
}


class ArmController:
    def __init__(self):
        self._arm_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
        try:
            from gripper_controller import GripperController
            self._gripper = GripperController()
        except Exception:
            self._gripper = None
        self._joint_names = (
            [f"left_arm_joint_{i}" for i in range(7)] +
            [f"right_arm_joint_{i}" for i in range(7)]
        )

    def move_to_pose(self, pose_name=None, positions=None, duration=1.0):
        if pose_name and pose_name in DEFAULT_ARM_POSES:
            positions = DEFAULT_ARM_POSES[pose_name]
        if positions is None:
            positions = [0.0] * 14
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.name = self._joint_names
        msg.position = list(positions)
        self._arm_pub.publish(msg)
        rospy.sleep(duration)

    def move_arm(self, arm="right", positions=None, duration=1.0):
        if positions is None:
            positions = [0.0] * 7
        if len(positions) != 7:
            rospy.logwarn("Arm positions must have 7 elements")
            return
        full_positions = [0.0] * 14
        if arm == "right":
            full_positions[7:] = positions
        else:
            full_positions[:7] = positions
        self.move_to_pose(positions=full_positions, duration=duration)

    def reach_to_position(self, target, arm="right", approach_height=0.3, duration=2.0):
        if arm == "right":
            pre_grasp = [
                0.8, target[1] * 0.5, 0.0, -1.5, 0.0, 0.0, 0.0
            ]
            grasp = [
                0.8, target[1] * 0.5, target[2] * 0.5, -1.5, 0.0, 0.0, 0.0
            ]
        else:
            pre_grasp = [
                0.8, -target[1] * 0.5, 0.0, 1.5, 0.0, 0.0, 0.0
            ]
            grasp = [
                0.8, -target[1] * 0.5, target[2] * 0.5, 1.5, 0.0, 0.0, 0.0
            ]

        self.move_arm(arm, pre_grasp, duration=1.0)
        self.move_arm(arm, grasp, duration=duration * 0.5)
        if self._gripper:
            self._gripper.close_grippers()
        rospy.sleep(0.5)
        self.move_arm(arm, pre_grasp, duration=duration * 0.5)

    def open_gripper(self, left=True, right=True):
        if self._gripper:
            self._gripper.open_grippers()

    def close_gripper(self, left=True, right=True):
        if self._gripper:
            self._gripper.close_grippers()

    def set_gripper_position(self, left=255, right=255):
        if self._gripper:
            self._gripper.set_gripper_position(left=left, right=right)

    def pickup_object(self, pick_position, arm="right", place_position=None):
        self.move_to_pose("home", duration=0.5)
        self.open_gripper()
        self.reach_to_position(pick_position, arm=arm)
        self.close_gripper()
        rospy.sleep(0.5)
        self.move_to_pose("up", duration=1.0)
        if place_position is not None:
            self.reach_to_position(place_position, arm=arm)
            self.open_gripper()
            rospy.sleep(0.5)
        self.move_to_pose("home", duration=1.0)

    def perform_grasp_sequence(self, pick_pos, place_pos, arm="right"):
        self.pickup_object(pick_pos, arm=arm, place_position=place_pos)
