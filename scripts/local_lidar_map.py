#!/usr/bin/env python3

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points

try:
    import tf2_ros

    HAS_TF2 = True
except Exception:  # pragma: no cover - optional in minimal dev env
    HAS_TF2 = False


@dataclass
class Clearance:
    front: float
    left_front: float
    right_front: float
    left: float
    right: float
    rear: float
    has_data: bool


def _quat_xyzw_to_rotation_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    """Unit quaternion x,y,z,w to 3x3 rotation matrix."""
    xx, yy, zz = qx * qx, qy * qy, qz * qz
    xy, xz, yz = qx * qy, qx * qz, qy * qz
    wx, wy, wz = qw * qx, qw * qy, qw * qz
    return np.array(
        [
            [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
            [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
            [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
        ],
        dtype=np.float64,
    )


class LocalLidarMapper:
    """Small robot-centric lidar map for reactive navigation.

    Simulation publishes /lidar/points in the lidar sensor frame by default (see
    craic_simulator lidar_mid360_node). When tf2 reports base_link<-lidar, points
    are transformed into robot_frame before sector split to match cmd_vel axes.
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
        self.robot_frame = rospy.get_param("~robot_frame", "base_link")
        self.use_tf = rospy.get_param("~lidar_use_tf", True)
        self._ema_alpha = float(rospy.get_param("~lidar_clearance_alpha", 0.4))
        self._turn_margin = float(rospy.get_param("~lidar_turn_margin", 0.22))

        self._lock = threading.Lock()
        self._latest_cloud = None
        self._sub = rospy.Subscriber(topic, PointCloud2, self._cloud_callback)

        self._tf_buffer = None
        self._tf_listener = None
        if HAS_TF2 and self.use_tf:
            self._tf_buffer = tf2_ros.Buffer(rospy.Duration(120.0))
            self._tf_listener = tf2_ros.TransformListener(self._tf_buffer)
            rospy.loginfo(
                "LocalLidarMapper: tf2 robot_frame=%s (cloud->robot when possible)",
                self.robot_frame,
            )
        else:
            rospy.logwarn(
                "LocalLidarMapper: tf unavailable or lidar_use_tf=false; raw cloud frame axes used",
            )

        self._warned_tf = False
        self._ema_state: Optional[Clearance] = None

    def _cloud_callback(self, msg):
        with self._lock:
            self._latest_cloud = msg

    def _transform_pts_to_robot_frame(
        self, cloud: PointCloud2, pts: np.ndarray
    ) -> Optional[np.ndarray]:
        if cloud.header.frame_id == self.robot_frame or not self.use_tf:
            return pts
        if self._tf_buffer is None:
            return pts

        ts = rospy.Time(0)
        try:
            t = self._tf_buffer.lookup_transform(self.robot_frame, cloud.header.frame_id, ts)
        except Exception:
            if not self._warned_tf:
                rospy.logwarn_throttle(
                    30.0,
                    "LocalLidarMapper: TF %s<- %s unavailable; lidar reactive nav uses sensor frame.",
                    self.robot_frame,
                    cloud.header.frame_id,
                )
                self._warned_tf = True
            return pts

        qw = float(t.transform.rotation.w)
        qx = float(t.transform.rotation.x)
        qy = float(t.transform.rotation.y)
        qz = float(t.transform.rotation.z)
        R = _quat_xyzw_to_rotation_matrix(qx, qy, qz, qw)
        tr = np.array(
            [
                float(t.transform.translation.x),
                float(t.transform.translation.y),
                float(t.transform.translation.z),
            ],
            dtype=np.float64,
        )
        return (R @ pts.T).T + tr

    def get_points(self, max_points: int = 3000) -> List[Tuple[float, float, float]]:
        with self._lock:
            cloud = self._latest_cloud
        if cloud is None:
            return []

        raw_list: List[Tuple[float, float, float]] = []
        for i, p in enumerate(read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if i >= max_points:
                break
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            distance_xy = math.hypot(x, y)
            if 0.12 <= distance_xy <= self.max_range and -0.25 <= z <= 1.25:
                raw_list.append((x, y, z))

        if not raw_list:
            return []

        arr = np.asarray(raw_list, dtype=np.float64)
        tf_arr = self._transform_pts_to_robot_frame(cloud, arr[:, :3])
        if tf_arr is None:
            return raw_list

        out: List[Tuple[float, float, float]] = []
        for i in range(tf_arr.shape[0]):
            x, y, z = float(tf_arr[i, 0]), float(tf_arr[i, 1]), float(tf_arr[i, 2])
            distance = math.hypot(x, y)
            if 0.12 <= distance <= self.max_range and -0.25 <= z <= 1.25:
                out.append((x, y, z))
        return out if out else raw_list

    def _clearance_raw(self, points: List[Tuple[float, float, float]]) -> Clearance:
        distances: Dict[str, float] = {name: self.max_range for name in self.SECTORS}
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

    def _ema_smooth(self, c: Clearance) -> Clearance:
        if not c.has_data or self._ema_alpha <= 0.0:
            return c
        p = self._ema_state
        if p is None or not p.has_data:
            self._ema_state = c
            return c
        alpha = max(0.0, min(1.0, self._ema_alpha))

        def b(a: float, b_: float):
            return (1.0 - alpha) * a + alpha * b_

        sm = Clearance(
            front=b(p.front, c.front),
            left_front=b(p.left_front, c.left_front),
            right_front=b(p.right_front, c.right_front),
            left=b(p.left, c.left),
            right=b(p.right, c.right),
            rear=b(p.rear, c.rear),
            has_data=True,
        )
        self._ema_state = sm
        return sm

    def clearance(self) -> Clearance:
        points = self.get_points()
        raw = self._clearance_raw(points)
        return self._ema_smooth(raw)

    def raw_clearance(self) -> Clearance:
        """Unsmoothed clearance for emergency braking (no EMA lag)."""
        points = self.get_points()
        return self._clearance_raw(points)

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
        m = self._turn_margin
        if left_score > right_score + m:
            return 1.0
        if right_score > left_score + m:
            return -1.0
        return 0.0

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
        err = max(-0.4, min(0.4, 0.5 * (c.right - c.left)))
        m = max(0.02, min(0.2, self._turn_margin))
        if abs(err) <= m:
            return 0.0
        return err

    @property
    def has_data(self) -> bool:
        with self._lock:
            return self._latest_cloud is not None

    # ========== 墙/锥桶 判别与拐角检测 ==========

    def _sector_filtered_points(
        self,
        angle_min: float,
        angle_max: float,
        points: List[Tuple[float, float, float]] = None,
    ) -> List[Tuple[float, float, float]]:
        """Extract points whose xy-angle falls inside [angle_min, angle_max]."""
        if points is None:
            points = self.get_points()
        result: List[Tuple[float, float, float]] = []
        for x, y, z in points:
            angle = math.atan2(y, x)
            if angle_min <= angle <= angle_max:
                result.append((x, y, z))
        return result

    @staticmethod
    def _distance_stats(
        pts: List[Tuple[float, float, float]],
    ) -> Tuple[int, float, float, float]:
        """Return (count, min_dist, mean_dist, std_dev) for a point list."""
        n = len(pts)
        if n == 0:
            return 0, float("inf"), 0.0, 0.0
        dists = [math.hypot(x, y) for x, y, _z in pts]
        mean_d = sum(dists) / n
        var_d = sum((d - mean_d) ** 2 for d in dists) / n
        return n, min(dists), mean_d, math.sqrt(var_d)

    @staticmethod
    def _cluster_by_distance(
        pts: List[Tuple[float, float, float]], radius: float
    ) -> List[List[Tuple[float, float, float]]]:
        """BFS clustering: points within *radius* of each other form a cluster."""
        if not pts:
            return []
        n = len(pts)
        unvisited = set(range(n))
        clusters: List[List[Tuple[float, float, float]]] = []
        for i in range(n):
            if i not in unvisited:
                continue
            cluster = [pts[i]]
            unvisited.discard(i)
            queue = [i]
            while queue:
                ci = queue.pop(0)
                cx, cy, _ = pts[ci]
                for j in list(unvisited):
                    px, py, _ = pts[j]
                    if (cx - px) ** 2 + (cy - py) ** 2 <= radius * radius:
                        cluster.append(pts[j])
                        queue.append(j)
                        unvisited.discard(j)
            clusters.append(cluster)
        return clusters

    def left_wall_info(self) -> Tuple[float, float, float, float]:
        """Analyze left side (18°–120°) for wall-like structure.

        Returns (wall_score, min_distance, mean_distance, variance).
        wall_score ∈ [0,1]: 1 = perfect contiguous wall, 0 = sparse/open space.
        """
        # combine left_front + left sectors
        pts_left = self._sector_filtered_points(math.radians(18), math.radians(120))
        count, min_d, mean_d, std_d = self._distance_stats(pts_left)
        if count < 8:
            return 0.0, self.max_range, self.max_range, 0.0
        count_factor = min(1.0, count / 55.0)
        variance_factor = 1.0 / (1.0 + std_d * 4.5)
        wall_score = count_factor * variance_factor
        return wall_score, min_d, mean_d, float(std_d * std_d)

    def right_cone_info(
        self,
        cone_diameter: float = 0.15,
        cone_min_points: int = 5,
        sparsity_threshold: float = 0.35,
    ) -> Tuple[float, int, float]:
        """Analyze right side (−120°–−18°) for discrete cone clusters.

        Returns (closest_cone_distance, num_cone_clusters, right_variance).
        Falls back to sparsity-based detection when clustering yields none.
        """
        pts_right = self._sector_filtered_points(math.radians(-120), math.radians(-18))
        if len(pts_right) < cone_min_points:
            return self.max_range, 0, 0.0

        clusters = self._cluster_by_distance(pts_right, cone_diameter)
        cones = [c for c in clusters if len(c) >= cone_min_points]

        if cones:
            min_dists = []
            for c in cones:
                cdists = [math.hypot(x, y) for x, y, _z in c]
                min_dists.append(min(cdists))
            closest = min(min_dists)
            _cnt, _min_d, _mean_d, std_d = self._distance_stats(pts_right)
            return closest, len(cones), float(std_d * std_d)

        # Cluster method failed — fall back to sparsity-based detection
        sp_score = self._sparsity_score_for_points(pts_right)
        if sp_score > sparsity_threshold:
            count, min_d, _mean_d, std_d = self._distance_stats(pts_right)
            est_cones = max(1, count // 3)
            return min_d, est_cones, float(std_d * std_d)

        return self.max_range, 0, 0.0

    def wall_score_left(self) -> float:
        """0-1 score: how wall-like is the left side."""
        score, _, _, _ = self.left_wall_info()
        return score

    def wall_score_right(self) -> float:
        """0-1 score: how wall-like is the right side (symmetric of left_wall_info)."""
        pts_right = self._sector_filtered_points(math.radians(-120), math.radians(-18))
        count, _, _, std_d = self._distance_stats(pts_right)
        if count < 8:
            return 0.0
        count_factor = min(1.0, count / 55.0)
        variance_factor = 1.0 / (1.0 + std_d * 4.5)
        return count_factor * variance_factor

    def corner_detected(
        self, threshold: float = 2.5
    ) -> bool:
        """Right-turn corner: left wall has vanished (both left and left_front open)."""
        c = self.clearance()
        if not c.has_data:
            return False
        return c.left > threshold and c.left_front > threshold * 0.75

    def direction_judge(
        self, wall_threshold: float = 0.5, cone_diameter: float = 0.15, cone_min_points: int = 5
    ) -> float:
        """Use wall/cone asymmetry to judge correct forward direction.

        Returns:
            1.0  – correct orientation (left=wall, right=cones)
           -1.0  – facing backwards (right=wall, left=cones)
            0.0  – uncertain
        """
        left_score = self.wall_score_left()
        right_score = self.wall_score_right()
        cone_dist_right, num_cones_right, _ = self.right_cone_info(cone_diameter, cone_min_points)

        left_is_wall = left_score > wall_threshold
        right_has_cones = num_cones_right > 0
        if left_is_wall and right_has_cones:
            return 1.0

        right_is_wall = right_score > wall_threshold
        pt_left = self._sector_filtered_points(math.radians(18), math.radians(120))
        clusters_left = self._cluster_by_distance(pt_left, cone_diameter)
        cones_left = [c for c in clusters_left if len(c) >= cone_min_points]
        left_has_cones = len(cones_left) > 0

        if right_is_wall and left_has_cones:
            return -1.0

        return 0.0

    def right_wall_info(self) -> Tuple[float, float, float, float]:
        """Analyze right side (−120°–−18°) for wall-like structure.

        Returns (wall_score, min_distance, mean_distance, variance).
        wall_score ∈ [0,1]: 1 = perfect contiguous wall, 0 = sparse/open space.
        """
        pts_right = self._sector_filtered_points(math.radians(-120), math.radians(-18))
        count, min_d, mean_d, std_d = self._distance_stats(pts_right)
        if count < 8:
            return 0.0, self.max_range, self.max_range, 0.0
        count_factor = min(1.0, count / 55.0)
        variance_factor = 1.0 / (1.0 + std_d * 4.5)
        wall_score = count_factor * variance_factor
        return wall_score, min_d, mean_d, float(std_d * std_d)

    def left_cone_info(
        self,
        cone_diameter: float = 0.15,
        cone_min_points: int = 5,
        sparsity_threshold: float = 0.35,
    ) -> Tuple[float, int, float]:
        """Analyze left side (18°–120°) for discrete cone clusters.

        Returns (closest_cone_distance, num_cone_clusters, left_variance).
        Falls back to sparsity-based detection when clustering yields none.
        """
        pts_left = self._sector_filtered_points(math.radians(18), math.radians(120))
        if len(pts_left) < cone_min_points:
            return self.max_range, 0, 0.0

        clusters = self._cluster_by_distance(pts_left, cone_diameter)
        cones = [c for c in clusters if len(c) >= cone_min_points]

        if cones:
            min_dists = []
            for c in cones:
                cdists = [math.hypot(x, y) for x, y, _z in c]
                min_dists.append(min(cdists))
            closest = min(min_dists)
            _cnt, _min_d, _mean_d, std_d = self._distance_stats(pts_left)
            return closest, len(cones), float(std_d * std_d)

        # Cluster method failed — fall back to sparsity-based detection
        sp_score = self._sparsity_score_for_points(pts_left)
        if sp_score > sparsity_threshold:
            count, min_d, _mean_d, std_d = self._distance_stats(pts_left)
            est_cones = max(1, count // 3)
            return min_d, est_cones, float(std_d * std_d)

        return self.max_range, 0, 0.0

    def left_corner_detected(self, threshold: float = 2.5) -> bool:
        """Left-turn corner: right wall has vanished (both right and right_front open)."""
        c = self.clearance()
        if not c.has_data:
            return False
        return c.right > threshold and c.right_front > threshold * 0.75

    def sparsity_score(self, sector_name: str) -> float:
        """Sparsity metric: 0.0 = dense/wall-like, 1.0 = sparse/cone-like.

        Based on point density per degree of arc.
        """
        sector_map = {
            'front': (math.radians(-18), math.radians(18)),
            'left_front': (math.radians(18), math.radians(68)),
            'right_front': (math.radians(-68), math.radians(-18)),
            'left': (math.radians(68), math.radians(120)),
            'right': (math.radians(-120), math.radians(-68)),
        }
        if sector_name not in sector_map:
            return 0.0
        lo, hi = sector_map[sector_name]
        pts = self._sector_filtered_points(lo, hi)
        return self._sparsity_score_for_points(pts)

    @staticmethod
    def _sparsity_score_for_points(
        pts: List[Tuple[float, float, float]],
    ) -> float:
        """Compute sparsity from a point list: 0.0 = dense, 1.0 = sparse."""
        n = len(pts)
        if n < 3:
            return 1.0
        density = n / 50.0
        if density >= 2.0:
            return 0.0
        if density <= 0.3:
            return 1.0
        return 1.0 - (density - 0.3) / 1.7
