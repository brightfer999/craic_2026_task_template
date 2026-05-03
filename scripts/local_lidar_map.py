#!/usr/bin/env python3

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points


@dataclass
class Clearance:
    front: float
    left_front: float
    right_front: float
    left: float
    right: float
    rear: float
    has_data: bool


class LocalLidarMapper:
    """Small robot-centric lidar map for reactive navigation.

    The mapper deliberately stays local: every point is interpreted in the
    current robot frame and converted into coarse sector clearance values.
    """

    SECTORS = {
        "front": (-math.radians(18), math.radians(18)),
        "left_front": (math.radians(18), math.radians(68)),
        "right_front": (-math.radians(68), -math.radians(18)),
        "left": (math.radians(68), math.radians(120)),
        "right": (-math.radians(120), -math.radians(68)),
        "rear": (math.radians(150), math.pi),
    }

    def __init__(self, topic: str = "/lidar/points", max_range: float = 4.5):
        self.max_range = max_range
        self._lock = threading.Lock()
        self._latest_cloud = None
        self._sub = rospy.Subscriber(topic, PointCloud2, self._cloud_callback)

    def _cloud_callback(self, msg):
        with self._lock:
            self._latest_cloud = msg

    def get_points(self, max_points: int = 3000) -> List[Tuple[float, float, float]]:
        with self._lock:
            cloud = self._latest_cloud
        if cloud is None:
            return []

        points = []
        for i, p in enumerate(read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if i >= max_points:
                break
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            distance = math.hypot(x, y)
            if 0.12 <= distance <= self.max_range and -0.25 <= z <= 1.25:
                points.append((x, y, z))
        return points

    def clearance(self) -> Clearance:
        distances: Dict[str, float] = {name: self.max_range for name in self.SECTORS}
        points = self.get_points()
        for x, y, _z in points:
            angle = math.atan2(y, x)
            distance = math.hypot(x, y)
            for name, (low, high) in self.SECTORS.items():
                if name == "rear":
                    in_sector = angle >= low or angle <= -low
                else:
                    in_sector = low <= angle <= high
                if in_sector and distance < distances[name]:
                    distances[name] = distance

        return Clearance(
            front=distances["front"],
            left_front=distances["left_front"],
            right_front=distances["right_front"],
            left=distances["left"],
            right=distances["right"],
            rear=distances["rear"],
            has_data=bool(points),
        )

    def front_clear(self, threshold: float = 0.75) -> bool:
        c = self.clearance()
        return (not c.has_data) or c.front >= threshold

    def preferred_turn(self) -> float:
        c = self.clearance()
        if not c.has_data:
            return 0.0
        if c.front > 1.1:
            return 0.0
        left_score = c.left_front + 0.5 * c.left
        right_score = c.right_front + 0.5 * c.right
        return 1.0 if left_score >= right_score else -1.0

    def corridor_error(self) -> float:
        c = self.clearance()
        if not c.has_data:
            return 0.0
        if c.left >= self.max_range and c.right >= self.max_range:
            return 0.0
        if c.left >= self.max_range:
            return -0.25
        if c.right >= self.max_range:
            return 0.25
        return max(-0.4, min(0.4, 0.5 * (c.right - c.left)))

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._latest_cloud is not None
