#!/usr/bin/env python3

import rospy
import time
from sensor_msgs.msg import JointState
from cv_bridge import CvBridge
import cv2
import numpy as np
from typing import Optional, Tuple, List
from collections import deque


class ButtonDetector:
    COLOR_RANGES = {
        "red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
        "green": (np.array([36, 50, 70]), np.array([86, 255, 255])),
        "blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    }

    def __init__(self, camera_topic="/cam_h/color/image_raw/compressed"):
        self._bridge = CvBridge()
        self._latest_frame = None
        self._camera_topic = camera_topic
        self._sub = None
        try:
            from sensor_msgs.msg import CompressedImage
            self._sub = rospy.Subscriber(camera_topic, CompressedImage, self._image_callback)
        except Exception:
            pass

    def _image_callback(self, msg):
        try:
            self._latest_frame = self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            try:
                self._latest_frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception:
                pass

    def detect_button(self, color_name="red") -> Optional[Tuple[int, int, float]]:
        if self._latest_frame is None or color_name not in self.COLOR_RANGES:
            return None

        hsv = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.COLOR_RANGES[color_name]
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:
            return None

        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None

        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        radius = np.sqrt(cv2.contourArea(largest) / np.pi)
        return cx, cy, radius

    def is_lit(self, color_name="red", threshold=0.15) -> Optional[bool]:
        if self._latest_frame is None or color_name not in self.COLOR_RANGES:
            return None

        hsv = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.COLOR_RANGES[color_name]
        mask = cv2.inRange(hsv, lower, upper)
        lit_ratio = np.count_nonzero(mask) / mask.size
        return lit_ratio > threshold

    @property
    def has_frame(self) -> bool:
        return self._latest_frame is not None


class ButtonPresser:
    GRIPPER_OPEN = 200
    GRIPPER_CLOSE = 0
    PRESS_DEPTH_OFFSET = 0.05

    def __init__(self):
        self._arm_pub = rospy.Publisher("/kuavo_arm_traj", JointState, queue_size=10)
        try:
            from gripper_controller import GripperController
            self._gripper = GripperController()
        except Exception:
            self._gripper = None

    def press_button_with_arm(self, arm="right", press_duration=1.0):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        if arm == "right":
            msg.position = [0.0] * 7 + [
                1.2, 0.4, 0.0, -1.57, 0.0, 0.0, 0.0
            ]
        else:
            msg.position = [
                1.2, -0.4, 0.0, 1.57, 0.0, 0.0, 0.0
            ] + [0.0] * 7
        self._arm_pub.publish(msg)
        rospy.sleep(1.5)

        if self._gripper:
            self._gripper.close_grippers()
        rospy.sleep(press_duration)

        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = [0.0] * 14
        self._arm_pub.publish(msg)
        rospy.sleep(1.0)

        if self._gripper:
            self._gripper.open_grippers()

    def extend_arm(self, arm="right", positions=None):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        if positions:
            msg.position = positions
        elif arm == "right":
            msg.position = [0.0] * 7 + [1.2, 0.4, 0.0, -1.57, 0.0, 0.0, 0.0]
        else:
            msg.position = [1.2, -0.4, 0.0, 1.57, 0.0, 0.0, 0.0] + [0.0] * 7
        self._arm_pub.publish(msg)

    def retract_arms(self):
        msg = JointState()
        msg.header.stamp = rospy.Time.now()
        msg.position = [0.0] * 14
        self._arm_pub.publish(msg)
