#!/usr/bin/env python3

import rospy
import math
from geometry_msgs.msg import Twist
from typing import Optional


class BasicMotion:
    def __init__(self):
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

    def move_forward(self, speed=0.3, duration=None):
        return self._send_cmd(linear_x=speed, duration=duration)

    def move_backward(self, speed=0.2, duration=None):
        return self._send_cmd(linear_x=-speed, duration=duration)

    def strafe_left(self, speed=0.15, duration=None):
        return self._send_cmd(linear_y=speed, duration=duration)

    def strafe_right(self, speed=0.15, duration=None):
        return self._send_cmd(linear_y=-speed, duration=duration)

    def turn_left(self, angular=0.5, duration=None):
        return self._send_cmd(angular_z=angular, duration=duration)

    def turn_right(self, angular=0.5, duration=None):
        return self._send_cmd(angular_z=-angular, duration=duration)

    def stop(self):
        self._send_cmd()

    def move_with_velocity(self, linear_x=0.0, linear_y=0.0, linear_z=0.0,
                           angular_x=0.0, angular_y=0.0, angular_z=0.0,
                           duration=None):
        return self._send_cmd(
            linear_x=linear_x, linear_y=linear_y, linear_z=linear_z,
            angular_x=angular_x, angular_y=angular_y, angular_z=angular_z,
            duration=duration
        )

    def rotate_to_yaw(self, target_yaw, current_yaw, angular_speed=0.4, tolerance=0.05):
        error = self._angle_diff(target_yaw, current_yaw)
        while abs(error) > tolerance:
            direction = 1.0 if error > 0 else -1.0
            self._send_cmd(angular_z=direction * angular_speed, duration=0.05)
            current_yaw += direction * angular_speed * 0.05
            error = self._angle_diff(target_yaw, current_yaw)
        self.stop()

    def move_to_target(self, target_x, target_y, current_x, current_y, current_yaw,
                       forward_speed=0.3, angular_speed=0.5, pos_tolerance=0.1):
        dx = target_x - current_x
        dy = target_y - current_y
        target_yaw = math.atan2(dy, dx)
        self.rotate_to_yaw(target_yaw, current_yaw, angular_speed)
        dist = math.hypot(dx, dy)
        travel_time = dist / forward_speed
        self.move_forward(forward_speed, duration=travel_time)
        self.stop()

    def _send_cmd(self, linear_x=0.0, linear_y=0.0, linear_z=0.0,
                  angular_x=0.0, angular_y=0.0, angular_z=0.0,
                  duration=None):
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = linear_y
        cmd.linear.z = linear_z
        cmd.angular.x = angular_x
        cmd.angular.y = angular_y
        cmd.angular.z = angular_z
        self._cmd_pub.publish(cmd)
        if duration is not None:
            rospy.sleep(duration)
            self.stop()

    @staticmethod
    def _angle_diff(a, b):
        diff = a - b
        return (diff + math.pi) % (2.0 * math.pi) - math.pi
