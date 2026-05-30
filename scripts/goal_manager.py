#!/usr/bin/env python3.8
"""Local goal generation without absolute world coordinates.

Generates short-range targets ("1.5 m ahead, biased toward traversable
centerline") purely from lidar sectors, obstacle positions, wall estimates,
and visited-memory cost.  No world waypoints are ever stored.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

from utils.math_utils import clamp, normalize_angle


@dataclass
class LocalGoal:
    """A local target expressed in the robot body frame."""

    x: float  # forward  [m]
    y: float  # lateral  [m]
    distance: float  # euclidean [m]
    angle: float  # heading relative to robot forward [rad]
    quality: float  # 0.0 – 1.0, how traversable the chosen direction is
    source: str  # "centerline" | "gap" | "fallback"


class GoalManager:
    """Generates local goals from lidar perception only."""

    def __init__(self):
        import rospy

        # Tunable parameters (overridable via ROS params for Phase 4 optimization)
        self.default_lookahead = 1.5  # m, nominal goal distance
        self.max_lookahead = 2.5  # m, extended when very open
        self.min_lookahead = 0.6  # m, reduced when tight
        self.centerline_sector_half_width = math.radians(35)  # ±35° search window
        self.traversability_min_clearance = float(
            rospy.get_param("~traversability_min_clearance", 0.50)
        )
        self.side_bias_gain = float(
            rospy.get_param("~goal_side_bias_gain", 0.35)
        )
        self.quality_decay_near_obstacle = 0.30  # per metre inside danger zone

        # Internal state -----------------------------------------------------
        self._last_goal: Optional[LocalGoal] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        clearance,  # Clearance dataclass
        obstacles: list,  # List[LocalObstacle]
        walls: list,  # List[WallEstimate]
        pose,  # PoseEstimate
        corridor_bias: float = 0.0,
        preferred_direction: float = 1.0,
        lidar_memory=None,  # LocalLidarMapper for visited-cost queries
    ) -> LocalGoal:
        """Produce one local goal from the current perception snapshot."""

        if not clearance.has_data:
            return self._fallback_goal(preferred_direction)

        # 1. Score narrow angular sectors for traversability -----------------
        sector_scores = self._score_traversable_sectors(clearance, obstacles, walls)

        # 2. Find the best continuous traversable region ---------------------
        best_angle, quality = self._find_centerline(sector_scores)

        # 3. Determine lookahead distance from clearance in that direction ---
        lookahead = self._compute_lookahead(best_angle, clearance, obstacles, walls)

        # 4. Apply corridor bias and memory penalty --------------------------
        best_angle = self._apply_bias_and_memory(
            best_angle, corridor_bias, preferred_direction, pose, lidar_memory
        )

        # 5. Build goal ------------------------------------------------------
        goal = LocalGoal(
            x=lookahead * math.cos(best_angle),
            y=lookahead * math.sin(best_angle),
            distance=lookahead,
            angle=best_angle,
            quality=clamp(quality, 0.0, 1.0),
            source="centerline" if quality > 0.35 else "gap",
        )
        self._last_goal = goal
        return goal

    # ------------------------------------------------------------------
    # Sector scoring
    # ------------------------------------------------------------------

    def _score_traversable_sectors(
        self, clearance, obstacles: list, walls: list
    ) -> List[float]:
        """Return a list of traversability scores for angular bins."""
        n_bins = 36  # 10° resolution
        scores = [1.0] * n_bins

        for i in range(n_bins):
            angle = math.radians(i * 10 - 175)  # [-175, 175] in 10° steps

            # --- Lidar sector clearance ---
            sector_dist = self._clearance_at_angle(clearance, angle)
            if sector_dist < self.traversability_min_clearance:
                scores[i] *= max(0.0, sector_dist / self.traversability_min_clearance)

            # --- Obstacle proximity ---
            for obs in obstacles:
                d = self._min_distance_to_obstacle_at_angle(angle, obs)
                if d < 0.30:
                    scores[i] *= 0.05
                elif d < 0.40:
                    scores[i] *= 0.30
                elif d < 0.65:
                    scores[i] *= 0.70

            # --- Wall penalty ---
            for wall in walls:
                if self._wall_blocks_angle(wall, angle):
                    scores[i] *= max(0.0, 1.0 - wall.score * 0.9)

            scores[i] = clamp(scores[i], 0.0, 1.0)

        return scores

    # ------------------------------------------------------------------
    # Centerline extraction
    # ------------------------------------------------------------------

    def _find_centerline(self, sector_scores: List[float]):
        """Find the continuous angular region with highest mean score."""
        n = len(sector_scores)
        best_angle = 0.0
        best_quality = -1.0
        best_angle_abs = float("inf")

        # Sliding window of ~70° (7 bins at 10° each)
        window = 7
        for i in range(n):
            region = [sector_scores[(i + j) % n] for j in range(window)]
            mean_score = sum(region) / window
            # Center angle of the window
            mid_bin = i + window // 2
            angle = math.radians(mid_bin * 10 - 175)
            angle = normalize_angle(angle)
            angle_abs = abs(angle)
            if (
                mean_score > best_quality + 1e-6
                or (abs(mean_score - best_quality) <= 1e-6 and angle_abs < best_angle_abs)
            ):
                best_quality = mean_score
                best_angle = angle
                best_angle_abs = angle_abs

        # If no region is clearly traversable, fall back to straight ahead
        if best_quality < 0.15:
            return 0.0, 0.0

        return best_angle, best_quality

    # ------------------------------------------------------------------
    # Lookahead
    # ------------------------------------------------------------------

    def _compute_lookahead(
        self, angle: float, clearance, obstacles: list, walls: list
    ) -> float:
        """How far ahead the goal can safely be placed."""
        sector_dist = self._clearance_at_angle(clearance, angle)

        # Clamp by nearest obstacle in that direction
        limit = sector_dist
        for obs in obstacles:
            d = self._min_distance_to_obstacle_at_angle(angle, obs)
            if d < limit:
                limit = d

        # Clamp between min/max
        if limit < 0.40:
            return self.min_lookahead
        if limit < 0.80:
            return max(self.min_lookahead, limit * 0.85)
        return clamp(limit * 0.65, self.default_lookahead, self.max_lookahead)

    # ------------------------------------------------------------------
    # Bias and memory
    # ------------------------------------------------------------------

    def _apply_bias_and_memory(
        self,
        angle: float,
        corridor_bias: float,
        preferred_direction: float,
        pose,
        lidar_memory,
    ) -> float:
        """Nudge the goal angle with corridor bias and visited-cost check."""
        # Corridor bias: right-wall-following prefers a slight rightward offset
        angle += corridor_bias * self.side_bias_gain

        # If reverse, flip the forward direction
        if preferred_direction < 0:
            angle = normalize_angle(angle + math.pi)

        # Check visited memory at a few candidate offsets
        if lidar_memory is not None and getattr(pose, "valid", False):
            best_angle = angle
            best_cost = float("inf")
            for offset in (-0.15, 0.0, 0.15):
                cand = normalize_angle(angle + offset)
                cost = 0.0
                for d in (0.5, 1.0, 1.5):
                    gx = pose.x + d * math.cos(pose.yaw + cand)
                    gy = pose.y + d * math.sin(pose.yaw + cand)
                    cost += lidar_memory.memory_cost_at(gx, gy)
                if cost < best_cost:
                    best_cost = cost
                    best_angle = cand
            angle = best_angle

        return clamp(angle, math.radians(-75), math.radians(75))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clearance_at_angle(clearance, angle: float) -> float:
        """Return the EMA clearance distance at a given heading."""
        deg = math.degrees(angle)
        if -18 <= deg <= 18:
            return clearance.front
        if 18 < deg <= 68:
            return clearance.left_front
        if -68 <= deg < -18:
            return clearance.right_front
        if 68 < deg <= 120:
            return clearance.left
        if -120 <= deg < -68:
            return clearance.right
        return clearance.rear

    @staticmethod
    def _min_distance_to_obstacle_at_angle(angle: float, obs) -> float:
        """Projected forward distance from robot to the obstacle at a heading."""
        ca = math.cos(angle)
        sa = math.sin(angle)
        forward = obs.x * ca + obs.y * sa
        lateral = abs(-obs.x * sa + obs.y * ca)
        if forward <= 0:
            return float("inf")
        return max(0.0, forward - obs.radius * 1.3)

    @staticmethod
    def _wall_blocks_angle(wall, angle: float) -> bool:
        """True when a wall estimate lies inside the angle cone."""
        if wall.side == "front" and abs(angle) < math.radians(20):
            return True
        if wall.side == "left" and angle > math.radians(5):
            return True
        if wall.side == "right" and angle < math.radians(-5):
            return True
        return False

    def _fallback_goal(self, preferred_direction: float) -> LocalGoal:
        """Straight-ahead fallback when lidar has no data."""
        d = 1.0 if preferred_direction > 0 else -1.0
        return LocalGoal(
            x=d,
            y=0.0,
            distance=abs(d),
            angle=0.0,
            quality=0.0,
            source="fallback",
        )
