"""
感知避障模块 — 基于官方 LidarMapper + ControllerManager 模板的扩展。

在 LidarMapper 的 6 扇区 clearance 基础上增加:
  - 锥桶检测 (BFS 聚类, 跨度和半径过滤)
  - 墙壁检测 (PCA 线拟合, 评分阈值)
  - 间隙查找 (角度分箱, 最佳通行方向)
  - 场景分类 (clear/front_blocked/left_blocked/right_blocked/corridor 等)
  - 安全速度计算 (VFH 风格 180° 决策)

设计原则:
  - 继承 skills/lidar_mapper.py 的 LidarMapper 作为感知基类
  - 组合 skills/controller_manager.py 的 ControllerManager 作为控制接口
  - 所有算法基于几何处理，不依赖深度学习
  - 参数通过 ROS param server 动态调整
"""

import math
import threading
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points

from .controller_manager import ControllerManager
from .lidar_mapper import Clearance, LidarMapper


# ============================================================================
# 数据类定义
# ============================================================================


@dataclass
class ConeDetection:
    """锥桶检测结果。"""
    x: float
    y: float
    distance: float
    radius: float
    point_count: int
    side: str  # "left" 或 "right"


@dataclass
class WallDetection:
    """墙壁检测结果。"""
    side: str
    score: float         # 0~1, 越高越像实心墙
    min_distance: float
    mean_distance: float
    angle: float         # 墙壁朝向角 (rad)
    length: float        # 墙壁拟合长度
    point_count: int


@dataclass
class GapTarget:
    """间隙目标。"""
    has_gap: bool
    angle: float         # 目标方向角 (rad)
    width: float         # 间隙宽度 (rad)
    clearance: float     # 间隙内最小距离
    front_clearance: float


@dataclass
class Scenario:
    """场景分类结果。"""
    name: str            # clear / front_blocked / left_blocked / right_blocked
                         # both_blocked / corridor / narrow_passage
    front_danger: bool
    left_danger: bool
    right_danger: bool
    cone_count: int
    wall_detected: bool
    confidence: float    # 0~1


# ============================================================================
# ObstacleDetector — 继承 LidarMapper 扩展感知能力
# ============================================================================


class ObstacleDetector(LidarMapper):
    """增强激光雷达感知器。

    继承 LidarMapper 的 6 扇区 clearance 计算，
    增加锥桶/墙壁检测、间隙查找和场景分类。
    """

    def __init__(self, topic: str = "/lidar/points", max_range: float = 4.5):
        super().__init__(topic=topic, max_range=max_range)

        # ---- 锥桶检测参数 ----
        self._cone_diameter = float(rospy.get_param("~cone_diameter", 0.15))
        self._cone_min_points = int(rospy.get_param("~cone_min_points", 3))
        self._cone_max_distance = float(rospy.get_param("~cone_max_distance", 2.6))

        # ---- 墙壁检测参数 ----
        self._wall_min_points = int(rospy.get_param("~wall_min_points", 10))
        self._wall_score_threshold = float(rospy.get_param("~wall_score_threshold", 0.20))

        # ---- 场景分类阈值 ----
        self._front_stop = float(rospy.get_param("~front_stop_distance", 0.40))
        self._front_slow = float(rospy.get_param("~front_slow_distance", 0.70))
        self._side_safe = float(rospy.get_param("~side_safe_distance", 0.42))

        # ---- 状态 ----
        self._last_cones: List[ConeDetection] = []
        self._last_walls: List[WallDetection] = []
        self._wall_sectors: List[str] = []

    # ------------------------------------------------------------------
    # 锥桶检测
    # ------------------------------------------------------------------

    def detect_cones(
        self,
        angle_range: Tuple[float, float] = (-2.18, 2.18),  # ±125°
    ) -> List[ConeDetection]:
        """BFS 聚类检测离散锥桶。

        1. 提取角度范围内的前向点云
        2. 用 cone_diameter 做 BFS 聚类
        3. 过滤 span > 0.55m 或 radius > 0.32m 的聚类
        4. 按 y 坐标分为 left/right
        """
        points = self.get_points()
        filtered = []
        for x, y, z in points:
            d = math.hypot(x, y)
            if d < 0.18 or d > self._cone_max_distance:
                continue
            angle = math.atan2(y, x)
            if angle_range[0] <= angle <= angle_range[1]:
                filtered.append((x, y, z))

        if len(filtered) < self._cone_min_points:
            self._last_cones = []
            return []

        clusters = self._bfs_cluster(filtered, self._cone_diameter * 1.6)
        cones = []
        for cluster in clusters:
            if len(cluster) < self._cone_min_points:
                continue
            cx, cy, dist, radius, span = self._cluster_shape(cluster)
            if span > 0.55 or radius > 0.32:
                continue
            side = "left" if cy >= 0.0 else "right"
            cones.append(ConeDetection(
                x=cx, y=cy, distance=dist, radius=max(radius, self._cone_diameter * 0.5),
                point_count=len(cluster), side=side,
            ))

        cones.sort(key=lambda c: c.distance)
        self._last_cones = cones
        return cones

    # ------------------------------------------------------------------
    # 墙壁检测
    # ------------------------------------------------------------------

    def detect_walls(self) -> List[WallDetection]:
        """PCA 线拟合检测墙壁结构。

        在 left (25°–125°), right (−125°–−25°), front (±35°) 三个扇区
        分别做 PCA 线拟合，计算 wall score。
        """
        sector_defs = [
            ("left", math.radians(25), math.radians(125)),
            ("right", math.radians(-125), math.radians(-25)),
            ("front", math.radians(-35), math.radians(35)),
        ]

        all_points = self.get_points()
        walls = []
        wall_sectors = []

        for side, lo, hi in sector_defs:
            pts = [(x, y, z) for x, y, z in all_points
                   if lo <= math.atan2(y, x) <= hi
                   and math.hypot(x, y) <= self.max_range]

            n, min_d, mean_d, std_d = self._distance_stats(pts)
            if n < self._wall_min_points:
                continue

            angle, length, residual = self._pca_fit_line(pts)
            count_factor = min(1.0, n / 60.0)
            length_factor = min(1.0, length / 0.9)
            residual_factor = 1.0 / (1.0 + residual * 18.0 + std_d * 0.6)
            score = count_factor * length_factor * residual_factor

            if score < self._wall_score_threshold:
                continue

            walls.append(WallDetection(
                side=side, score=score, min_distance=min_d,
                mean_distance=mean_d, angle=angle,
                length=length, point_count=n,
            ))
            wall_sectors.append(side)

        self._last_walls = walls
        self._wall_sectors = wall_sectors
        return walls

    # ------------------------------------------------------------------
    # 锥桶检测 (排除墙壁扇区)
    # ------------------------------------------------------------------

    def detect_cones_excluding_walls(self) -> List[ConeDetection]:
        """检测锥桶，自动排除已识别为墙壁的区域。"""
        # 先确保 walls 是最新的
        if not self._last_walls:
            self.detect_walls()

        wall_sector_angles = {
            "left": (math.radians(25), math.radians(125)),
            "right": (math.radians(-125), math.radians(-25)),
            "front": (math.radians(-35), math.radians(35)),
        }

        points = self.get_points()
        filtered = []
        for x, y, z in points:
            d = math.hypot(x, y)
            if d < 0.18 or d > self._cone_max_distance:
                continue
            angle = math.atan2(y, x)
            if abs(angle) > math.radians(125):
                continue

            # 排除墙壁扇区内的点
            in_wall = False
            for ws in self._wall_sectors:
                if ws in wall_sector_angles:
                    w_lo, w_hi = wall_sector_angles[ws]
                    if w_lo <= angle <= w_hi:
                        in_wall = True
                        break
            if in_wall:
                continue

            filtered.append((x, y, z))

        if len(filtered) < self._cone_min_points:
            self._last_cones = []
            return []

        clusters = self._bfs_cluster(filtered, self._cone_diameter * 1.6)
        cones = []
        for cluster in clusters:
            if len(cluster) < self._cone_min_points:
                continue
            cx, cy, dist, radius, span = self._cluster_shape(cluster)
            if span > 0.55 or radius > 0.32:
                continue
            side = "left" if cy >= 0.0 else "right"
            cones.append(ConeDetection(
                x=cx, y=cy, distance=dist,
                radius=max(radius, self._cone_diameter * 0.5),
                point_count=len(cluster), side=side,
            ))

        cones.sort(key=lambda c: c.distance)
        self._last_cones = cones
        return cones

    # ------------------------------------------------------------------
    # 间隙查找
    # ------------------------------------------------------------------

    def find_best_gap(
        self,
        angle_limit: float = math.radians(75),
        bin_count: int = 61,
        lookahead: float = 2.2,
        safety_radius: float = 0.38,
        preferred_angle: float = 0.0,
    ) -> GapTarget:
        """在 ±angle_limit 范围内寻找最佳通行间隙。

        将前方扇形空间分成 bin_count 个角度分箱，
        统计每个分箱的 clearance，找到最大连续自由区域。
        """
        points = [
            p for p in self.get_points()
            if 0.18 <= math.hypot(p[0], p[1]) <= lookahead
            and abs(math.atan2(p[1], p[0])) <= angle_limit
        ]

        front_clearance = float(self.max_range)
        for x, y, _z in points:
            if abs(math.atan2(y, x)) <= math.radians(15):
                front_clearance = min(front_clearance, math.hypot(x, y))

        if bin_count < 5:
            bin_count = 5
        angles = np.linspace(-angle_limit, angle_limit, bin_count)
        clearances = [lookahead] * bin_count

        for x, y, _z in points:
            dist = math.hypot(x, y)
            if dist <= 0.001:
                continue
            angle = math.atan2(y, x)
            spread = min(math.radians(28), math.asin(min(0.95, safety_radius / dist)))
            for idx, ray_ang in enumerate(angles):
                if abs(ray_ang - angle) <= spread:
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
            return GapTarget(
                has_gap=False, angle=float(angles[idx]), width=0.0,
                clearance=float(clearances[idx]),
                front_clearance=float(front_clearance),
            )

        best = None
        best_score = -float("inf")
        for lo, hi in gaps:
            mid = (lo + hi) // 2
            mid_angle = float(angles[mid])
            width = float(angles[hi] - angles[lo])
            gap_clearance = min(clearances[lo:hi + 1])
            score = (
                width * 1.8
                + gap_clearance * 0.8
                + math.cos(mid_angle)
                - abs(mid_angle - preferred_angle) * 0.9
            )
            if score > best_score:
                best_score = score
                best = (mid_angle, width, gap_clearance)

        angle, width, gap_clearance = best
        return GapTarget(
            has_gap=True, angle=angle, width=width,
            clearance=gap_clearance, front_clearance=float(front_clearance),
        )

    # ------------------------------------------------------------------
    # 场景分类
    # ------------------------------------------------------------------

    def classify_scenario(self) -> Scenario:
        """根据 clearance + 锥桶/墙壁信息判断当前场景类型。

        Returns:
            Scenario 包含分类名、各方向危险标志和置信度。
        """
        c = self.clearance()
        if not c.has_data:
            return Scenario(
                name="no_data", front_danger=False,
                left_danger=False, right_danger=False,
                cone_count=0, wall_detected=False, confidence=0.0,
            )

        cones = self._last_cones
        walls = self._last_walls
        if not walls:
            walls = self.detect_walls()
        if not cones:
            cones = self.detect_cones_excluding_walls()

        front_danger = c.front < self._front_stop
        front_slow = c.front < self._front_slow
        left_danger = min(c.left_front, c.left) < self._side_safe
        right_danger = min(c.right_front, c.right) < self._side_safe

        cone_count = len(cones)
        wall_detected = len(walls) > 0

        left_wall_score = sum(w.score for w in walls if w.side == "left")
        right_wall_score = sum(w.score for w in walls if w.side == "right")

        # 场景判定 (优先级从高到低):
        if front_danger and left_danger and right_danger:
            name = "both_blocked"
            confidence = 0.95
        elif front_danger and left_danger:
            name = "front_left_blocked"
            confidence = 0.90
        elif front_danger and right_danger:
            name = "front_right_blocked"
            confidence = 0.90
        elif front_danger:
            name = "front_blocked"
            confidence = 0.85
        elif left_danger and right_danger:
            name = "both_sides_blocked"
            confidence = 0.80
        elif left_danger:
            name = "left_blocked"
            confidence = 0.80
        elif right_danger:
            name = "right_blocked"
            confidence = 0.80
        elif left_wall_score > 0.3 and right_wall_score > 0.3:
            name = "corridor"
            confidence = 0.75
        elif left_wall_score > 0.3 or right_wall_score > 0.3:
            name = "one_wall"
            confidence = 0.70
        elif front_slow:
            name = "front_slow"
            confidence = 0.65
        else:
            name = "clear"
            confidence = 0.60

        return Scenario(
            name=name, front_danger=front_danger,
            left_danger=left_danger, right_danger=right_danger,
            cone_count=cone_count, wall_detected=wall_detected,
            confidence=confidence,
        )

    @property
    def all_detections(self):
        """获取一帧内所有检测结果的快照。"""
        return {
            "clearance": self.clearance(),
            "cones": list(self._last_cones),
            "walls": list(self._last_walls),
        }

    # ==================================================================
    # 内部辅助方法
    # ==================================================================

    @staticmethod
    def _bfs_cluster(pts, radius):
        if not pts:
            return []
        n = len(pts)
        unvisited = set(range(n))
        clusters = []
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
                    if (cx - px) ** 2 + (cy - py) ** 2 <= radius ** 2:
                        cluster.append(pts[j])
                        queue.append(j)
                        unvisited.discard(j)
            clusters.append(cluster)
        return clusters

    @staticmethod
    def _cluster_shape(pts):
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        cx = sum(xs) / len(xs)
        cy = sum(ys) / len(ys)
        radius = max(math.hypot(x - cx, y - cy) for x, y, _z in pts)
        span = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        return cx, cy, math.hypot(cx, cy), radius, span

    @staticmethod
    def _distance_stats(pts):
        n = len(pts)
        if n == 0:
            return 0, float("inf"), 0.0, 0.0
        dists = [math.hypot(x, y) for x, y, _z in pts]
        mean_d = sum(dists) / n
        var_d = sum((d - mean_d) ** 2 for d in dists) / n
        return n, min(dists), mean_d, math.sqrt(var_d)

    @staticmethod
    def _pca_fit_line(pts):
        if len(pts) < 2:
            return 0.0, 0.0, float("inf")
        arr = np.asarray([(x, y) for x, y, _z in pts], dtype=np.float64)
        center = np.mean(arr, axis=0)
        centered = arr - center
        _, vecs = np.linalg.eigh(centered.T @ centered)
        direction = vecs[:, -1]
        projections = centered @ direction
        length = float(np.max(projections) - np.min(projections))
        normal = np.array([-direction[1], direction[0]], dtype=np.float64)
        residual = float(np.mean(np.abs(centered @ normal)))
        angle = math.atan2(float(direction[1]), float(direction[0]))
        return angle, length, residual


# ============================================================================
# AvoidanceController — 基于感知结果生成安全速度指令
# ============================================================================


class AvoidanceController:
    """基于 ObstacleDetector 的自动避障控制器。

    感知 → 决策 → 速度指令 的完整闭环。
    组合 ControllerManager 将速度指令发布到 /cmd_vel。
    """

    def __init__(
        self,
        detector: ObstacleDetector,
        controller: Optional[ControllerManager] = None,
    ):
        self._detector = detector
        self._controller = controller

        # ---- 速度参数 ----
        self._v_max = float(rospy.get_param("~v_max", 0.20))
        self._w_max = float(rospy.get_param("~w_max", 0.45))
        self._v_slow = float(rospy.get_param("~v_slow", 0.08))
        self._v_medium = float(rospy.get_param("~v_medium", 0.12))
        self._v_fast = float(rospy.get_param("~v_fast", 0.20))

        # ---- 控制参数 ----
        self._front_stop = float(rospy.get_param("~front_stop_distance", 0.40))
        self._front_slow = float(rospy.get_param("~front_slow_distance", 0.70))
        self._side_safe = float(rospy.get_param("~side_safe_distance", 0.42))
        self._cone_safe = float(rospy.get_param("~cone_safe_distance", 0.50))
        self._wall_follow_dist = float(rospy.get_param("~wall_follow_distance", 0.80))
        self._wall_kp_lat = float(rospy.get_param("~wall_kp_lat", 0.22))
        self._wall_kp_ang = float(rospy.get_param("~wall_kp_ang", 0.20))
        self._gap_lookahead = float(rospy.get_param("~obstacle_gap_lookahead", 2.2))
        self._gap_safety_radius = float(rospy.get_param("~obstacle_gap_safety_radius", 0.40))

        # ---- 累计转向锁 ----
        self._initial_yaw: Optional[float] = None
        self._max_cumulative_turn = math.radians(126.0)
        self._turn_locked = False

        # ---- 日志 ----
        self._last_log_time = rospy.Time.now()

    # ------------------------------------------------------------------
    # 决策表
    # ------------------------------------------------------------------

    def compute_safe_velocity(
        self,
        clearance: Clearance,
        scenario: Scenario,
    ) -> Tuple[float, float, float]:
        """根据感知结果计算安全速度 (vx, vy, wz)。

        决策表:
            clear            → 匀速前进
            front_blocked    → 停车，原地转向更开阔侧
            left_blocked     → 右前方偏移 (vy>0, wz>0)
            right_blocked    → 左前方偏移 (vy<0, wz<0)
            both_blocked     → 减速蠕行，等待间隙
            corridor         → 居中 PID 跟随
            one_wall         → 沿墙跟随 (右侧优先)
            front_slow       → 减速前进
            no_data          → 原地等待
        """
        name = scenario.name
        cones = self._detector._last_cones
        walls = self._detector._last_walls

        if name == "no_data":
            return 0.0, 0.0, 0.0

        # ---- gap 查找 (用于前方有障时) ----
        gap = self._detector.find_best_gap(
            lookahead=self._gap_lookahead,
            safety_radius=self._gap_safety_radius,
        )

        # ---- 原始 clearance (用于紧急避障) ----
        raw = clearance  # 简化: 复用 clearance

        raw_front_min = min(raw.front, raw.left_front, raw.right_front) if raw.has_data else 99.0
        raw_left_min = min(raw.left_front, raw.left) if raw.has_data else 99.0
        raw_right_min = min(raw.right_front, raw.right) if raw.has_data else 99.0

        # ================================================================
        # 紧急刹车
        # ================================================================
        if raw_front_min < self._front_stop * 0.65:
            left_space = raw_left_min
            right_space = raw_right_min
            wz = self._w_max * (-1.0 if right_space >= left_space else 1.0)
            rospy.logwarn_throttle(0.8, "  紧急刹车: front=%.2fm", raw_front_min)
            return 0.0, 0.0, wz

        # ================================================================
        # 场景分支
        # ================================================================
        vx, vy, wz = 0.0, 0.0, 0.0

        if name == "clear":
            vx = self._v_fast
            wz = 0.0

        elif name == "front_blocked":
            # 前方有障 → 停车 + 向开阔侧原地转向
            left_space = raw_left_min if not scenario.left_danger else 0.0
            right_space = raw_right_min if not scenario.right_danger else 0.0
            if gap.has_gap:
                vx = self._v_slow
                wz = gap.angle * 0.8
            else:
                wz = self._w_max * (1.0 if left_space > right_space else -1.0)

        elif name in ("front_left_blocked", "front_right_blocked"):
            # 前方 + 一侧有障 → 向安全侧偏移
            if gap.has_gap:
                vx = self._v_slow
                wz = gap.angle * 0.7
                vy = math.sin(gap.angle) * gap.clearance * 0.3
            else:
                # 向右后转
                vx = 0.02
                wz = -self._w_max if name == "front_left_blocked" else self._w_max

        elif name == "left_blocked":
            # 左侧有障 → 向右前方偏移
            vx = self._v_medium
            avoid = (self._side_safe - raw_left_min) / self._side_safe
            vy = 0.08 * avoid
            wz = 0.18 * avoid
            # 锥桶额外排斥力
            for cone in cones:
                if cone.side == "left" and cone.distance < self._cone_safe:
                    push = (self._cone_safe - cone.distance) / self._cone_safe
                    wz += 0.15 * push
                    vy += 0.06 * push

        elif name == "right_blocked":
            # 右侧有障 → 向左前方偏移
            vx = self._v_medium
            avoid = (self._side_safe - raw_right_min) / self._side_safe
            vy = -0.08 * avoid
            wz = -0.18 * avoid
            for cone in cones:
                if cone.side == "right" and cone.distance < self._cone_safe:
                    push = (self._cone_safe - cone.distance) / self._cone_safe
                    wz -= 0.15 * push
                    vy -= 0.06 * push

        elif name == "both_sides_blocked":
            # 两侧都有障 → 减速蠕行 + 向更开阔侧微调
            vx = self._v_slow
            left_clear = raw_left_min if not scenario.left_danger else 0.2
            right_clear = raw_right_min if not scenario.right_danger else 0.2
            if gap.has_gap:
                wz = gap.angle * 0.5
            else:
                wz = 0.10 * (1.0 if left_clear > right_clear else -1.0)

        elif name == "both_blocked":
            # 三面都有障 → 原地转向 + 低速试探
            if gap.has_gap:
                vx = self._v_slow * 0.5
                wz = gap.angle * 0.6
            else:
                # 向侧面空间更大的一侧转
                wz = self._w_max * (1.0 if raw_left_min > raw_right_min else -1.0)

        elif name == "corridor":
            # 两侧都有墙 → 居中 PID 跟随
            vx = self._v_medium
            wall_err = raw.right - self._wall_follow_dist
            vy = -np.clip(self._wall_kp_lat * wall_err, -0.12, 0.12)
            wz = np.clip(-self._wall_kp_ang * wall_err, -0.22, 0.22)

        elif name == "one_wall":
            # 单侧有墙 → 沿墙跟随 (右侧优先)
            vx = self._v_medium
            right_wall = any(w.side == "right" for w in walls)
            left_wall = any(w.side == "left" for w in walls)
            if right_wall:
                wall_err = raw.right - self._wall_follow_dist
            elif left_wall:
                wall_err = self._wall_follow_dist - raw.left
            else:
                wall_err = 0.0
            vy = np.clip(-self._wall_kp_lat * wall_err, -0.12, 0.12)
            wz = np.clip(-self._wall_kp_ang * wall_err, -0.18, 0.18)

        elif name == "front_slow":
            # 前方稍近但不危险 → 减速前进
            slow_factor = max(0.3, (raw.front - self._front_stop) / (self._front_slow - self._front_stop))
            vx = self._v_medium * slow_factor

        # ================================================================
        # 安全钳制
        # ================================================================
        vx = np.clip(vx, 0.0, self._v_max)
        vy = np.clip(vy, -0.18, 0.18)
        wz = np.clip(wz, -self._w_max, self._w_max)

        # 如果累计转向角过大，锁死转向
        if self._turn_locked:
            wz = 0.0
            vx = min(vx, self._v_slow)

        return vx, vy, wz

    # ------------------------------------------------------------------
    # 单步控制
    # ------------------------------------------------------------------

    def step(self) -> Twist:
        """一帧的感知 → 决策 → 速度指令。"""
        clearance = self._detector.clearance()
        if not clearance.has_data:
            rospy.logwarn_throttle(2.0, "  无激光雷达数据，停止等待...")
            return Twist()

        # 更新检测
        self._detector.detect_walls()
        self._detector.detect_cones_excluding_walls()

        # 场景分类
        scenario = self._detector.classify_scenario()

        # 计算速度
        vx, vy, wz = self.compute_safe_velocity(clearance, scenario)

        # 定期日志
        now = rospy.Time.now()
        if (now - self._last_log_time).to_sec() >= 0.5:
            self._log_state(scenario, vx, vy, wz)
            self._last_log_time = now

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = vy
        cmd.angular.z = wz
        return cmd

    def send_command(self):
        """执行一步感知-避障-控制，通过 ControllerManager 发布速度。"""
        cmd = self.step()
        if self._controller is not None:
            self._controller.send_velocity_command(
                linear_x=cmd.linear.x,
                linear_y=cmd.linear.y,
                angular_z=cmd.angular.z,
            )
        return cmd

    def stop(self):
        """紧急停止。"""
        if self._controller is not None:
            self._controller.stop_robot(0.5)
        else:
            cmd = Twist()
            pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
            pub.publish(cmd)
            rospy.sleep(0.3)
            pub.publish(cmd)

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _log_state(self, scenario, vx, vy, wz):
        c = self._detector.clearance()
        cones = self._detector._last_cones
        walls = self._detector._last_walls
        rospy.loginfo(
            "  [%s] v=(%.2f,%.2f,%.2f) | "
            "clear: f=%.2f lf=%.2f rf=%.2f l=%.2f r=%.2f | "
            "cones=%d walls=%s | conf=%.2f",
            scenario.name, vx, vy, wz,
            c.front, c.left_front, c.right_front, c.left, c.right,
            len(cones),
            ",".join(f"{w.side}:{w.score:.2f}" for w in walls[:3]) or "none",
            scenario.confidence,
        )
