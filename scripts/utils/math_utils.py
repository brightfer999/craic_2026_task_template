#!/usr/bin/env python3

import math
from collections import deque
from typing import Optional


def clamp(value, low, high):
    return max(low, min(value, high))


def normalize_angle(angle):
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def distance_2d(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def distance_3d(p1, p2):
    return math.sqrt(
        (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2 + (p1[2] - p2[2]) ** 2
    )


def quaternion_to_euler(qx, qy, qz, qw):
    sinr_cosp = 2.0 * (qw * qx + qy * qz)
    cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (qw * qy - qz * qx)
    pitch = math.asin(clamp(sinp, -1.0, 1.0))

    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return roll, pitch, yaw


def euler_to_quaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    qw = cr * cp * cy + sr * sp * sy
    qx = sr * cp * cy - cr * sp * sy
    qy = cr * sp * cy + sr * cp * sy
    qz = cr * cp * sy - sr * sp * cy
    return qx, qy, qz, qw


def linear_map(value, in_min, in_max, out_min, out_max):
    if in_min == in_max:
        return out_min
    return out_min + (value - in_min) * (out_max - out_min) / (in_max - in_min)


class MovingAverage:
    def __init__(self, window_size=10):
        self._window = deque(maxlen=window_size)

    def update(self, value):
        self._window.append(value)

    def get(self) -> Optional[float]:
        if not self._window:
            return None
        return sum(self._window) / len(self._window)

    def clear(self):
        self._window.clear()


moving_average = MovingAverage
