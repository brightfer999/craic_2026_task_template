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


@dataclass
class ConeCluster:
    x: float
    y: float
    distance: float
    radius: float
    point_count: int
    side: str


@dataclass
class WallEstimate:
    side: str
    score: float
    min_distance: float
    mean_distance: float
    angle: float
    length: float
    point_count: int


@dataclass
class GapTarget:
    has_gap: bool
    angle: float
    lateral_offset: float
    width: float
    clearance: float
    front_clearance: float


@dataclass
class ConfirmedCone:
    cone: ConeCluster
    confidence: float
    frames_seen: int


@dataclass
class ConfirmedWall:
    wall: WallEstimate
    confidence: float
    frames_seen: int


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

        # 多帧确认历史缓冲区
        self._history_size = int(rospy.get_param("~detection_history_size", 5))
        self._confirmation_threshold = int(rospy.get_param("~detection_confirmation_threshold", 3))
        self._cone_history: List[List[ConeCluster]] = []
        self._wall_history: List[List[WallEstimate]] = []
        self._wall_sectors_cache: List[str] = []

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

    @staticmethod
    def _cluster_shape(
        pts: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float, float, float]:
        """Return centroid x/y, radial distance, max radius and xy span."""
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radius = max(math.hypot(x - cx, y - cy) for x, y, _z in pts)
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return cx, cy, math.hypot(cx, cy), radius, span

    def detect_cones(
        self,
        cone_diameter: float = 0.15,
        cone_min_points: int = 3,
        angle_min: float = math.radians(-125),
        angle_max: float = math.radians(125),
        max_distance: float = 2.6,
        exclude_wall_sectors: Optional[List[str]] = None,
    ) -> List[ConeCluster]:
        """Detect compact discrete clusters that behave like traffic cones."""
        # 互斥分类：排除已被识别为墙壁的扇区
        wall_sector_angles = {
            "left": (math.radians(25), math.radians(125)),
            "right": (math.radians(-125), math.radians(-25)),
            "front": (math.radians(-35), math.radians(35)),
        }
        
        points = []
        for p in self._sector_filtered_points(angle_min, angle_max):
            if not (0.18 <= math.hypot(p[0], p[1]) <= max_distance):
                continue
            
            # 检查该点是否在墙壁扇区内
            if exclude_wall_sectors:
                point_angle = math.atan2(p[1], p[0])
                in_wall_sector = False
                for sector in exclude_wall_sectors:
                    if sector in wall_sector_angles:
                        lo, hi = wall_sector_angles[sector]
                        if lo <= point_angle <= hi:
                            in_wall_sector = True
                            break
                if in_wall_sector:
                    continue
            
            points.append(p)
        
        if len(points) < cone_min_points:
            return []
        if len(points) > 420:
            step = int(math.ceil(len(points) / 420.0))
            points = points[::step]

        cluster_radius = max(0.18, cone_diameter * 1.6)
        cones: List[ConeCluster] = []
        for cluster in self._cluster_by_distance(points, cluster_radius):
            if len(cluster) < cone_min_points:
                continue
            cx, cy, dist, radius, span = self._cluster_shape(cluster)
            if span > 0.55 or radius > 0.32:
                continue
            side = "left" if cy >= 0.0 else "right"
            cones.append(
                ConeCluster(
                    x=cx,
                    y=cy,
                    distance=dist,
                    radius=max(radius, cone_diameter * 0.5),
                    point_count=len(cluster),
                    side=side,
                )
            )
        cones.sort(key=lambda c: c.distance)
        return cones

    @staticmethod
    def _fit_line_estimate(
        pts: List[Tuple[float, float, float]],
    ) -> Tuple[float, float, float]:
        """PCA line estimate: heading angle, segment length and mean residual."""
        if len(pts) < 2:
            return 0.0, 0.0, float("inf")
        arr = np.asarray([(x, y) for x, y, _z in pts], dtype=np.float64)
        center = np.mean(arr, axis=0)
        centered = arr - center
        _vals, vecs = np.linalg.eigh(centered.T @ centered)
        direction = vecs[:, -1]
        projections = centered @ direction
        length = float(np.max(projections) - np.min(projections))
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residual = float(np.mean(np.abs(centered @ normal)))
        angle = math.atan2(float(direction[1]), float(direction[0]))
        return angle, length, residual

    def detect_walls(self) -> List[WallEstimate]:
        """Detect continuous side-wall structures and reject sparse cone clusters."""
        walls: List[WallEstimate] = []
        wall_sectors = []
        sectors = (
            ("left", math.radians(25), math.radians(125)),
            ("right", math.radians(-125), math.radians(-25)),
            ("front", math.radians(-35), math.radians(35)),
        )
        for side, lo, hi in sectors:
            pts = self._sector_filtered_points(lo, hi)
            pts = [p for p in pts if math.hypot(p[0], p[1]) <= self.max_range]
            count, min_d, mean_d, std_d = self._distance_stats(pts)
            if count < 10:
                continue
            angle, length, residual = self._fit_line_estimate(pts)
            count_factor = min(1.0, count / 60.0)
            length_factor = min(1.0, length / 0.9)
            residual_factor = 1.0 / (1.0 + residual * 18.0 + std_d * 0.6)
            score = count_factor * length_factor * residual_factor
            if score < 0.20:
                continue
            walls.append(
                WallEstimate(
                    side=side,
                    score=score,
                    min_distance=min_d,
                    mean_distance=mean_d,
                    angle=angle,
                    length=length,
                    point_count=count,
                )
            )
            wall_sectors.append(side)
        
        # 更新墙壁扇区缓存
        self._wall_sectors_cache = wall_sectors
        
        # 更新墙壁历史缓冲区
        self._wall_history.append(walls)
        if len(self._wall_history) > self._history_size:
            self._wall_history.pop(0)
        
        return walls

    def detect_cones_with_exclusion(
        self,
        cone_diameter: float = 0.15,
        cone_min_points: int = 3,
        angle_min: float = math.radians(-125),
        angle_max: float = math.radians(125),
        max_distance: float = 2.6,
    ) -> List[ConeCluster]:
        """检测锥桶，自动排除已被识别为墙壁的扇区。"""
        return self.detect_cones(
            cone_diameter=cone_diameter,
            cone_min_points=cone_min_points,
            angle_min=angle_min,
            angle_max=angle_max,
            max_distance=max_distance,
            exclude_wall_sectors=self._wall_sectors_cache,
        )

    def get_confirmed_cones(
        self,
        cone_diameter: float = 0.15,
        cone_min_points: int = 3,
        angle_min: float = math.radians(-125),
        angle_max: float = math.radians(125),
        max_distance: float = 2.6,
    ) -> List[ConfirmedCone]:
        """获取经过多帧确认的锥桶检测结果。"""
        current_cones = self.detect_cones_with_exclusion(
            cone_diameter=cone_diameter,
            cone_min_points=cone_min_points,
            angle_min=angle_min,
            angle_max=angle_max,
            max_distance=max_distance,
        )
        
        # 更新锥桶历史缓冲区
        self._cone_history.append(current_cones)
        if len(self._cone_history) > self._history_size:
            self._cone_history.pop(0)
        
        if len(self._cone_history) < self._confirmation_threshold:
            return []
        
        # 统计每个锥桶在历史帧中出现的次数
        confirmed_cones = []
        for cone in current_cones:
            frames_seen = 0
            for historical_cones in self._cone_history:
                for hist_cone in historical_cones:
                    # 检查是否是同一个锥桶（基于位置接近性）
                    distance = math.hypot(cone.x - hist_cone.x, cone.y - hist_cone.y)
                    if distance < 0.3:  # 30cm 内认为是同一个锥桶
                        frames_seen += 1
                        break
            
            if frames_seen >= self._confirmation_threshold:
                confidence = frames_seen / len(self._cone_history)
                confirmed_cones.append(ConfirmedCone(
                    cone=cone,
                    confidence=confidence,
                    frames_seen=frames_seen,
                ))
        
        return confirmed_cones

    def get_confirmed_walls(self) -> List[ConfirmedWall]:
        """获取经过多帧确认的墙壁检测结果。"""
        if len(self._wall_history) < self._confirmation_threshold:
            return []
        
        # 统计每个墙壁扇区在历史帧中出现的次数
        wall_counts = {}
        for historical_walls in self._wall_history:
            for wall in historical_walls:
                if wall.side not in wall_counts:
                    wall_counts[wall.side] = {
                        "count": 0,
                        "wall": wall,
                        "scores": [],
                    }
                wall_counts[wall.side]["count"] += 1
                wall_counts[wall.side]["scores"].append(wall.score)
        
        confirmed_walls = []
        for side, data in wall_counts.items():
            if data["count"] >= self._confirmation_threshold:
                confidence = data["count"] / len(self._wall_history)
                avg_score = sum(data["scores"]) / len(data["scores"])
                # 使用最新的墙壁数据，但用平均分数
                latest_wall = data["wall"]
                confirmed_wall = WallEstimate(
                    side=latest_wall.side,
                    score=avg_score,
                    min_distance=latest_wall.min_distance,
                    mean_distance=latest_wall.mean_distance,
                    angle=latest_wall.angle,
                    length=latest_wall.length,
                    point_count=latest_wall.point_count,
                )
                confirmed_walls.append(ConfirmedWall(
                    wall=confirmed_wall,
                    confidence=confidence,
                    frames_seen=data["count"],
                ))
        
        return confirmed_walls

    def find_best_gap(
        self,
        angle_limit: float = math.radians(75),
        bin_count: int = 61,
        lookahead: float = 2.2,
        safety_radius: float = 0.38,
        preferred_angle: float = 0.0,
    ) -> GapTarget:
        """Choose a forward free-space gap; cones and walls are both obstacles."""
        points = [
            p for p in self.get_points()
            if 0.18 <= math.hypot(p[0], p[1]) <= lookahead
            and -angle_limit <= math.atan2(p[1], p[0]) <= angle_limit
        ]
        front_clearance = self.max_range
        for x, y, _z in points:
            a = math.atan2(y, x)
            if abs(a) <= math.radians(15):
                front_clearance = min(front_clearance, math.hypot(x, y))

        if bin_count < 5:
            bin_count = 5
        angles = np.linspace(-angle_limit, angle_limit, bin_count)
        clearances = [lookahead for _ in range(bin_count)]

        for x, y, _z in points:
            dist = math.hypot(x, y)
            angle = math.atan2(y, x)
            if dist <= 0.001:
                continue
            spread = min(math.radians(28), math.asin(min(0.95, safety_radius / dist)))
            for idx, ray_angle in enumerate(angles):
                if abs(ray_angle - angle) <= spread:
                    clearances[idx] = min(clearances[idx], dist)

        free = [c >= max(0.55, safety_radius * 1.45) for c in clearances]
        gaps: List[Tuple[int, int]] = []
        start = None
        for idx, ok in enumerate(free):
            if ok and start is None:
                start = idx
            elif not ok and start is not None:
                gaps.append((start, idx - 1))
                start = None
        if start is not None:
            gaps.append((start, len(free) - 1))

        if not gaps:
            idx = int(np.argmax(clearances))
            angle = float(angles[idx])
            return GapTarget(False, angle, math.sin(angle) * clearances[idx], 0.0,
                             float(clearances[idx]), float(front_clearance))

        best = None
        best_score = -float("inf")
        for lo, hi in gaps:
            center_idx = (lo + hi) // 2
            center_angle = float(angles[center_idx])
            width = float(angles[hi] - angles[lo])
            clearance = min(clearances[lo:hi + 1])
            forward_score = math.cos(center_angle)
            preference_penalty = abs(center_angle - preferred_angle)
            score = width * 1.8 + clearance * 0.8 + forward_score - preference_penalty * 0.9
            if score > best_score:
                best_score = score
                best = (center_angle, width, clearance)

        angle, width, clearance = best
        return GapTarget(True, angle, math.sin(angle) * min(clearance, lookahead),
                         width, clearance, float(front_clearance))

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
            1.0  – correct orientation (right=wall, left=cones) for right-wall-following
           -1.0  – facing backwards (left=wall, right=cones)
            0.0  – uncertain
        """
        left_score = self.wall_score_left()
        right_score = self.wall_score_right()

        # Check right wall + left cones → correct direction
        right_is_wall = right_score > wall_threshold
        pt_left = self._sector_filtered_points(math.radians(18), math.radians(120))
        clusters_left = self._cluster_by_distance(pt_left, cone_diameter)
        cones_left = [c for c in clusters_left if len(c) >= cone_min_points]
        left_has_cones = len(cones_left) > 0

        if right_is_wall and left_has_cones:
            return 1.0

        # Check left wall + right cones → facing backwards
        left_is_wall = left_score > wall_threshold
        cone_dist_right, num_cones_right, _ = self.right_cone_info(cone_diameter, cone_min_points)
        right_has_cones = num_cones_right > 0

        if left_is_wall and right_has_cones:
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
