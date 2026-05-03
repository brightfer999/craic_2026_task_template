#!/usr/bin/env python3

import rospy
import math
import struct
import numpy as np
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points
from collections import defaultdict
from typing import Optional, List, Tuple


class ObstacleDetector:
    def __init__(self, safe_distance=0.5, min_distance=0.3, field_of_view=math.radians(180)):
        self.safe_distance = safe_distance
        self.min_distance = min_distance
        self.field_of_view = field_of_view
        self._latest_cloud = None
        self._sub = rospy.Subscriber("/lidar/points", PointCloud2, self._cloud_callback)

    def _cloud_callback(self, msg):
        self._latest_cloud = msg

    def get_obstacles(self) -> List[List[float]]:
        if self._latest_cloud is None:
            return []
        points = []
        for p in read_points(self._latest_cloud, field_names=("x", "y", "z"), skip_nans=True):
            dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            if dist <= self.safe_distance:
                points.append([p[0], p[1], p[2]])
        return points

    def check_front_clear(self, lookahead=0.5, angle_range=math.radians(30)) -> bool:
        cloud = self._latest_cloud
        if cloud is None:
            return True
        for p in read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            angle = abs(math.atan2(p[1], p[0]))
            if dist < lookahead and angle < angle_range and p[0] > 0:
                return False
        return True

    def find_best_direction(self) -> float:
        cloud = self._latest_cloud
        if cloud is None:
            return 0.0
        sector_cost = defaultdict(float)
        sector_count = defaultdict(int)
        for p in read_points(cloud, field_names=("x", "y", "z"), skip_nans=True):
            dist = math.sqrt(p[0]**2 + p[1]**2 + p[2]**2)
            if dist < self.safe_distance:
                angle = math.atan2(p[1], p[0])
                sector = int(round(angle / math.radians(10)))
                sector_cost[sector] += 1.0 / max(dist, 0.01)
                sector_count[sector] += 1
        if not sector_cost:
            return 0.0
        sectors = list(sector_cost.keys())
        sectors.sort(key=lambda s: sector_cost[s])
        best_sector = sectors[0]
        return best_sector * math.radians(10)

    def get_sector_distances(self, num_sectors=36) -> List[float]:
        cloud = self._latest_cloud
        if cloud is None:
            return [float("inf")] * num_sectors

        sector_min = [float("inf")] * num_sectors
        angle_per_sector = 2.0 * math.pi / num_sectors

        for p in read_points(cloud, field_names=("x", "y"), skip_nans=True):
            dist = math.hypot(p[0], p[1])
            angle = math.atan2(p[1], p[0]) + math.pi
            sector = int(angle / angle_per_sector) % num_sectors
            if dist < sector_min[sector]:
                sector_min[sector] = dist
        return sector_min

    @property
    def has_data(self) -> bool:
        return self._latest_cloud is not None
