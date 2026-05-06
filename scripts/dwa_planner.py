#!/usr/bin/env python3.8
"""DWA (Dynamic Window Approach) local planner for obstacle navigation.

Samples (v, w) pairs within the dynamic window, forward-simulates each
trajectory 1.5–2.5 s, scores them with a multi-term cost function, and
outputs the best cmd_vel.  Falls back to gap-follow when no trajectory
passes the safety threshold.
"""

import math
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import rospy
from geometry_msgs.msg import Twist

from goal_manager import GoalManager, LocalGoal
from utils.math_utils import clamp, normalize_angle


@dataclass
class DWAParams:
    """All tunable parameters for the DWA planner."""

    # Dynamic window bounds
    v_max: float = 0.22
    w_max: float = 0.50
    v_accel: float = 0.35  # m/s²
    w_accel: float = 0.80  # rad/s²
    v_samples: int = 14
    w_samples: int = 25

    # Trajectory simulation
    sim_dt: float = 0.10  # s
    sim_time_min: float = 1.5  # s
    sim_time_max: float = 2.5  # s
    sim_time_adaptive: bool = True  # shorter horizon when tight

    # Obstacle cost
    obstacle_danger_dist: float = 0.30  # m — strong penalty below this
    obstacle_caution_dist: float = 0.40  # m — cautious zone
    obstacle_danger_cost: float = 100.0
    obstacle_caution_cost: float = 3.5
    obstacle_close_decay: float = 0.85  # per-metre decay inside caution

    # Goal cost
    goal_heading_weight: float = 2.2
    goal_proximity_reward: float = 1.0  # per metre closer to goal
    forward_progress_weight: float = 1.15
    caution_turn_reward: float = 0.65
    danger_extra_cost: float = 4.0

    # Wall / cone clearance reward
    clearance_reward_gain: float = 0.35  # per metre away from walls

    # Visited memory penalty
    memory_cost_weight: float = 0.65
    memory_check_steps: int = 5  # check N evenly-spaced points

    # Oscillation penalty
    angular_oscillation_weight: float = 1.8  # per rad/s change

    # Backward / U-turn penalty
    backward_cost: float = 80.0
    uturn_angle_thresh: float = math.radians(110)
    uturn_cost: float = 50.0
    global_yaw_weight: float = 2.0
    global_yaw_soft_limit: float = math.radians(45)
    global_yaw_hard_limit: float = math.radians(95)

    # Scoring
    min_acceptable_score: float = -4.0  # below this → consider fallback

    # Speed ramping for cautious zones
    cautious_linear_max: float = 0.06  # max v when clearance 0.30–0.40
    danger_linear_max: float = 0.03  # max v when clearance < 0.30


    @classmethod
    def from_ros_params(cls):
        return cls(
            v_max=float(rospy.get_param("~dwa_v_max", rospy.get_param("~v_max", cls.v_max))),
            w_max=float(rospy.get_param("~dwa_w_max", rospy.get_param("~w_max", cls.w_max))),
            v_accel=float(rospy.get_param("~dwa_v_accel", cls.v_accel)),
            w_accel=float(rospy.get_param("~dwa_w_accel", cls.w_accel)),
            v_samples=int(rospy.get_param("~dwa_v_samples", cls.v_samples)),
            w_samples=int(rospy.get_param("~dwa_w_samples", cls.w_samples)),
            sim_dt=float(rospy.get_param("~dwa_sim_dt", cls.sim_dt)),
            sim_time_min=float(rospy.get_param("~dwa_sim_time_min", cls.sim_time_min)),
            sim_time_max=float(rospy.get_param("~dwa_sim_time_max", cls.sim_time_max)),
            sim_time_adaptive=bool(rospy.get_param("~dwa_sim_time_adaptive", cls.sim_time_adaptive)),
            obstacle_danger_dist=float(rospy.get_param("~obstacle_danger_dist", cls.obstacle_danger_dist)),
            obstacle_caution_dist=float(rospy.get_param("~obstacle_caution_dist", cls.obstacle_caution_dist)),
            obstacle_danger_cost=float(rospy.get_param("~dwa_obstacle_danger_cost", cls.obstacle_danger_cost)),
            obstacle_caution_cost=float(rospy.get_param("~dwa_obstacle_caution_cost", cls.obstacle_caution_cost)),
            obstacle_close_decay=float(rospy.get_param("~dwa_obstacle_close_decay", cls.obstacle_close_decay)),
            goal_heading_weight=float(rospy.get_param("~dwa_goal_heading_weight", cls.goal_heading_weight)),
            goal_proximity_reward=float(rospy.get_param("~dwa_goal_proximity_reward", cls.goal_proximity_reward)),
            forward_progress_weight=float(rospy.get_param("~dwa_forward_progress_weight", cls.forward_progress_weight)),
            caution_turn_reward=float(rospy.get_param("~dwa_caution_turn_reward", cls.caution_turn_reward)),
            danger_extra_cost=float(rospy.get_param("~dwa_danger_extra_cost", cls.danger_extra_cost)),
            clearance_reward_gain=float(rospy.get_param("~dwa_clearance_reward_gain", cls.clearance_reward_gain)),
            memory_cost_weight=float(rospy.get_param("~dwa_memory_cost_weight", cls.memory_cost_weight)),
            memory_check_steps=int(rospy.get_param("~dwa_memory_check_steps", cls.memory_check_steps)),
            angular_oscillation_weight=float(rospy.get_param("~dwa_oscillation_weight", cls.angular_oscillation_weight)),
            backward_cost=float(rospy.get_param("~dwa_backward_cost", cls.backward_cost)),
            uturn_angle_thresh=math.radians(
                float(rospy.get_param("~dwa_uturn_angle_thresh_deg", math.degrees(cls.uturn_angle_thresh)))
            ),
            uturn_cost=float(rospy.get_param("~dwa_uturn_cost", cls.uturn_cost)),
            global_yaw_weight=float(rospy.get_param("~dwa_global_yaw_weight", cls.global_yaw_weight)),
            global_yaw_soft_limit=math.radians(
                float(rospy.get_param("~dwa_global_yaw_soft_limit_deg", math.degrees(cls.global_yaw_soft_limit)))
            ),
            global_yaw_hard_limit=math.radians(
                float(rospy.get_param("~dwa_global_yaw_hard_limit_deg", math.degrees(cls.global_yaw_hard_limit)))
            ),
            min_acceptable_score=float(rospy.get_param("~dwa_min_acceptable_score", cls.min_acceptable_score)),
            cautious_linear_max=float(rospy.get_param("~dwa_cautious_linear_max", cls.cautious_linear_max)),
            danger_linear_max=float(rospy.get_param("~dwa_danger_linear_max", cls.danger_linear_max)),
        )


@dataclass
class TrajectoryScore:
    v: float
    w: float
    total_cost: float
    min_clearance: float
    goal_alignment: float
    is_safe: bool


class DWAPlanner:
    """Dynamic Window Approach planner with gap-follow fallback."""

    def __init__(
        self,
        params: Optional[DWAParams] = None,
        lidar_mapper=None,  # LocalLidarMapper for memory queries
    ):
        self.params = params or DWAParams()
        self._lidar = lidar_mapper
        self._goal_manager = GoalManager()

        # Internal state
        self._prev_v = 0.0
        self._prev_w = 0.0
        self._last_cmd_time: Optional[float] = None
        self._oscillation_filter = 0.0  # EMA of angular changes
        self._frame_idx = 0

        # Fallback scoring (mirrors existing gap-follow logic)
        self._fallback_angles_deg = [-55, -40, -28, -16, 0, 16, 28, 40, 55]
        self._fallback_angles = [math.radians(a) for a in self._fallback_angles_deg]

        # Scoring diagnostics for logging (Phase 4)
        self._last_candidates: List[TrajectoryScore] = []
        self._last_selected: Optional[TrajectoryScore] = None
        self._last_was_fallback: bool = False
        self._last_fallback_score: Optional[TrajectoryScore] = None

        rospy.loginfo("DWAPlanner initialized v_max=%.2f w_max=%.2f",
                      self.params.v_max, self.params.w_max)

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def plan(
        self,
        clearance,           # Clearance dataclass
        obstacles: list,     # List[LocalObstacle]
        walls: list,         # List[WallEstimate]
        pose,                # PoseEstimate
        corridor_bias: float = 0.0,
        yaw_delta: float = 0.0,
        preferred_direction: float = 1.0,
    ) -> Tuple[Twist, TrajectoryScore, bool]:
        """Compute one DWA planning cycle.

        Returns:
            cmd:        Twist to publish
            score:      TrajectoryScore with diagnostics
            fallback:   True when gap-follow was used instead of DWA
        """
        self._frame_idx += 1

        # Generate local goal
        local_goal = self._goal_manager.generate(
            clearance=clearance,
            obstacles=obstacles,
            walls=walls,
            pose=pose,
            corridor_bias=corridor_bias,
            preferred_direction=preferred_direction,
            lidar_memory=self._lidar,
        )

        # Build dynamic window
        v_samples, w_samples = self._build_dynamic_window()

        # Score all trajectories
        best, candidates = self._score_trajectories(
            v_samples, w_samples, obstacles, walls, local_goal, pose, yaw_delta
        )
        self._last_candidates = candidates
        self._last_selected = best

        # Decide: keep DWA in control whenever a trajectory is physically safe.
        # Low reward should bias selection, not silently hand control back to the old path.
        use_fallback = not best.is_safe
        self._last_was_fallback = use_fallback

        if use_fallback:
            cmd, fallback_score = self._fallback_gap_follow(
                clearance, obstacles, walls, pose, yaw_delta, preferred_direction
            )
            self._last_fallback_score = fallback_score
            rospy.loginfo_throttle(
                1.0,
                "DWA fallback → gap-follow  v=%.2f w=%.2f  score=%.2f",
                cmd.linear.x, cmd.angular.z, fallback_score.total_cost,
            )
            return cmd, fallback_score, True

        # Build cmd_vel from best trajectory
        cmd = Twist()
        cmd.linear.x = clamp(best.v, 0.0, self.params.v_max)

        # Apply speed limits based on clearance
        if best.min_clearance < self.params.obstacle_danger_dist:
            cmd.linear.x = min(cmd.linear.x, self.params.danger_linear_max)
        elif best.min_clearance < self.params.obstacle_caution_dist:
            cmd.linear.x = min(cmd.linear.x, self.params.cautious_linear_max)

        cmd.angular.z = clamp(best.w, -self.params.w_max, self.params.w_max)
        cmd.linear.y = 0.0

        # Update state for next cycle
        dt = self._dt_since_last_cmd()
        self._update_oscillation_filter(best.w, dt)
        self._prev_v = cmd.linear.x
        self._prev_w = cmd.angular.z
        self._last_cmd_time = time.time()

        return cmd, best, False

    # ------------------------------------------------------------------
    # Dynamic window
    # ------------------------------------------------------------------

    def _build_dynamic_window(self):
        """Sample (v, w) pairs achievable given acceleration limits."""
        dt = max(self._dt_since_last_cmd(), 0.05)

        # Velocity bounds from acceleration
        v_low = max(0.0, self._prev_v - self.params.v_accel * dt)
        v_high = min(self.params.v_max, self._prev_v + self.params.v_accel * dt)

        w_low = max(-self.params.w_max, self._prev_w - self.params.w_accel * dt)
        w_high = min(self.params.w_max, self._prev_w + self.params.w_accel * dt)

        # Sample
        v_samples = [
            v_low + (v_high - v_low) * i / max(1, self.params.v_samples - 1)
            for i in range(self.params.v_samples)
        ]
        w_samples = [
            w_low + (w_high - w_low) * i / max(1, self.params.w_samples - 1)
            for i in range(self.params.w_samples)
        ]

        return v_samples, w_samples

    # ------------------------------------------------------------------
    # Trajectory scoring
    # ------------------------------------------------------------------

    def _score_trajectories(
        self,
        v_samples: List[float],
        w_samples: List[float],
        obstacles: list,
        walls: list,
        goal: LocalGoal,
        pose,
        yaw_delta: float = 0.0,
    ) -> Tuple[TrajectoryScore, List[TrajectoryScore]]:
        best = TrajectoryScore(
            v=0.0, w=0.0, total_cost=float("inf"),
            min_clearance=0.0, goal_alignment=0.0, is_safe=False,
        )
        all_scores: List[TrajectoryScore] = []

        for v in v_samples:
            for w in w_samples:
                score = self._evaluate_trajectory(v, w, obstacles, walls, goal, pose, yaw_delta)
                all_scores.append(score)
                if score.total_cost < best.total_cost:
                    best = score

        # Keep top-10 candidates for diagnostics
        all_scores.sort(key=lambda s: s.total_cost)
        top_n = all_scores[:10]

        return best, top_n

    def _evaluate_trajectory(
        self,
        v: float,
        w: float,
        obstacles: list,
        walls: list,
        goal: LocalGoal,
        pose,
        yaw_delta: float = 0.0,
    ) -> TrajectoryScore:
        p = self.params

        # Determine simulation horizon adaptively
        sim_time = p.sim_time_max
        if p.sim_time_adaptive:
            front_clear = self._front_clearance(obstacles)
            if front_clear < 0.50:
                sim_time = p.sim_time_min
            elif front_clear < 1.0:
                sim_time = p.sim_time_min + (front_clear - 0.5) / 0.5 * (
                    p.sim_time_max - p.sim_time_min
                )

        n_steps = max(3, int(sim_time / p.sim_dt))
        dt = sim_time / n_steps

        # --- Forward simulate ---
        min_clearance = float("inf")
        total_memory_cost = 0.0
        n_memory_checks = 0

        # For wall/cone clearance reward accumulation
        clearance_sum = 0.0
        clearance_count = 0

        for step in range(1, n_steps + 1):
            t = step * dt

            # Trajectory point in robot frame
            if abs(w) < 0.001:
                rx = v * t
                ry = 0.0
            else:
                rx = v / w * math.sin(w * t)
                ry = v / w * (1.0 - math.cos(w * t))

            # Obstacle clearance at this point
            step_clearance = self._clearance_at_point(rx, ry, obstacles)
            wall_dist = self._wall_distance_at_point(rx, ry, walls)
            min_clearance = min(min_clearance, step_clearance, wall_dist)

            # Wall proximity reward (we want to be away from walls)
            clearance_sum += wall_dist
            clearance_count += 1

            # Visited memory check (every few steps)
            if step % max(1, n_steps // p.memory_check_steps) == 0:
                if self._lidar is not None and getattr(pose, "valid", False):
                    wx = pose.x + rx * math.cos(pose.yaw) - ry * math.sin(pose.yaw)
                    wy = pose.y + rx * math.sin(pose.yaw) + ry * math.cos(pose.yaw)
                    total_memory_cost += self._lidar.memory_cost_at(wx, wy)
                    n_memory_checks += 1

        # --- Build cost ---
        cost = 0.0

        # 1. Obstacle clearance
        if min_clearance < p.obstacle_danger_dist:
            cost += p.obstacle_danger_cost * (1.0 + (p.obstacle_danger_dist - min_clearance) * 3.0)
        elif min_clearance < p.obstacle_caution_dist:
            cost += p.obstacle_caution_cost * (p.obstacle_caution_dist - min_clearance) / (
                p.obstacle_caution_dist - p.obstacle_danger_dist
            )

        # 2. Wall/cone clearance reward (negative cost = reward)
        avg_wall_dist = clearance_sum / max(1, clearance_count)
        cost -= p.clearance_reward_gain * min(avg_wall_dist, 2.0)

        # 3. Forward progress reward. This is what makes open space visibly move
        # faster while still letting clearance penalties dominate near obstacles.
        cost -= p.forward_progress_weight * v

        # Reward controlled avoidance in the 0.30-0.40 m band and punish
        # getting closer than 0.30 m, matching the training signal used offline.
        if 0.30 <= min_clearance <= 0.40 and abs(w) > 0.06:
            cost -= p.caution_turn_reward
        elif min_clearance < 0.30:
            cost += p.danger_extra_cost * (0.30 - min_clearance) / 0.30

        # 4. Goal progress
        # Final heading relative to goal direction
        final_theta = w * sim_time if abs(w) >= 0.001 else 0.0
        heading_to_goal = normalize_angle(goal.angle - final_theta)
        goal_alignment = max(0.0, math.cos(heading_to_goal))
        cost -= p.goal_heading_weight * goal_alignment

        # Goal proximity: prefer trajectories that end closer to goal
        if abs(w) < 0.001:
            end_x = v * sim_time
            end_y = 0.0
        else:
            end_x = v / w * math.sin(w * sim_time)
            end_y = v / w * (1.0 - math.cos(w * sim_time))
        dist_to_goal = math.hypot(end_x - goal.x, end_y - goal.y)
        cost += p.goal_proximity_reward * dist_to_goal

        # 5. Visited memory penalty
        if n_memory_checks > 0:
            cost += p.memory_cost_weight * total_memory_cost / n_memory_checks

        # 6. Angular oscillation penalty
        w_diff = abs(w - self._prev_w)
        cost += p.angular_oscillation_weight * w_diff * (1.0 + self._oscillation_filter)

        # 7. Backward penalty
        if v < 0.0:
            cost += p.backward_cost

        # 8. U-turn penalty: trajectory ends pointing backwards or too far sideways
        if abs(final_theta) > p.uturn_angle_thresh:
            cost += p.uturn_cost

        # 9. Stage-relative yaw penalty: each short trajectory can look safe
        # while the robot slowly accumulates a large turn. Keep the net heading
        # aligned with the stage start so local DWA cannot drift into a loop.
        final_global_yaw = abs(normalize_angle(yaw_delta + final_theta))
        if final_global_yaw > p.global_yaw_soft_limit:
            over = final_global_yaw - p.global_yaw_soft_limit
            span = max(0.05, p.global_yaw_hard_limit - p.global_yaw_soft_limit)
            cost += p.global_yaw_weight * (over / span) ** 2
        if final_global_yaw > p.global_yaw_hard_limit:
            cost += p.uturn_cost * (1.0 + final_global_yaw - p.global_yaw_hard_limit)

        # Safety check
        is_safe = (
            min_clearance >= p.obstacle_danger_dist * 0.85
            and cost < 50.0
        )

        return TrajectoryScore(
            v=v,
            w=w,
            total_cost=cost,
            min_clearance=min_clearance,
            goal_alignment=goal_alignment,
            is_safe=is_safe,
        )

    # ------------------------------------------------------------------
    # Fallback: gap-follow planner (preserved from existing logic)
    # ------------------------------------------------------------------

    def _fallback_gap_follow(
        self,
        clearance,
        obstacles: list,
        walls: list,
        pose,
        yaw_delta: float,
        preferred_direction: float = 1.0,
    ) -> Tuple[Twist, TrajectoryScore]:
        """Score candidate steering angles using the existing gap-based approach."""
        preferred_angle = clamp(-yaw_delta, math.radians(-35), math.radians(35))
        best_angle = preferred_angle
        best_score = -float("inf")
        best_clearance = 4.5

        for angle in self._fallback_angles:
            clr, collision_loss = self._gap_candidate_clearance(angle, obstacles)
            reward = 0.0
            loss = 0.0

            reward += 2.4 * max(0.0, math.cos(angle))
            reward += 1.3 * clamp((clr - 0.30) / 1.0, 0.0, 1.0)
            reward += 0.8 * clamp(
                1.0 - abs(angle - preferred_angle) / math.radians(75), 0.0, 1.0
            )

            if 0.30 <= clr <= 0.40 and abs(angle) > math.radians(10):
                reward += 0.9

            loss += collision_loss
            loss += 0.9 * abs(angle) / math.radians(75)
            loss += 0.6 * abs(normalize_angle(yaw_delta + angle))
            loss += self._gap_candidate_memory_loss(angle, pose)

            if math.cos(angle) < 0.45:
                loss += 3.0

            score = reward - loss
            if score > best_score:
                best_score = score
                best_angle = angle
                best_clearance = clr

        best_angle = clamp(best_angle, math.radians(-55), math.radians(55))

        # Determine linear speed from clearance
        if best_clearance < 0.30:
            vx = 0.035
        elif best_clearance < 0.40:
            vx = 0.06
        elif best_clearance < 0.75:
            vx = 0.10
        else:
            vx = min(self.params.v_max, 0.18)

        cmd = Twist()
        cmd.linear.x = vx * preferred_direction if preferred_direction > 0 else 0.0
        cmd.linear.y = clamp(0.32 * math.sin(best_angle), -0.18, 0.18)
        cmd.angular.z = clamp(0.95 * best_angle - 0.70 * yaw_delta, -0.35, 0.35)

        score = TrajectoryScore(
            v=cmd.linear.x,
            w=cmd.angular.z,
            total_cost=-best_score,  # convert reward→cost convention
            min_clearance=best_clearance,
            goal_alignment=math.cos(best_angle),
            is_safe=True,
        )
        return cmd, score

    def _gap_candidate_clearance(self, angle: float, obstacles: list) -> Tuple[float, float]:
        ca = math.cos(angle)
        sa = math.sin(angle)
        clearance = 4.5
        loss = 0.0
        safety_radius = 0.40
        lookahead = 2.2
        for obs in obstacles:
            forward = obs.x * ca + obs.y * sa
            lateral = abs(-obs.x * sa + obs.y * ca)
            if forward <= 0.0 or forward > lookahead:
                continue
            guard = safety_radius + obs.radius
            if lateral < guard:
                local_clear = max(0.0, forward - obs.radius)
                clearance = min(clearance, local_clear)
                risk = (guard - lateral) / max(0.05, guard)
                loss += (1.2 + 2.5 * risk) * obs.confidence
            elif lateral < guard + 0.25:
                loss += 0.45 * obs.confidence
        return clearance, loss

    def _gap_candidate_memory_loss(self, angle: float, pose) -> float:
        if self._lidar is None or not getattr(pose, "valid", False):
            return 0.0
        loss = 0.0
        for dist in (0.35, 0.70, 1.05, 1.40, 1.75):
            gx = pose.x + dist * math.cos(pose.yaw + angle)
            gy = pose.y + dist * math.sin(pose.yaw + angle)
            loss += self._lidar.memory_cost_at(gx, gy)
        return 0.35 * loss

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clearance_at_point(rx: float, ry: float, obstacles: list) -> float:
        """Minimum distance from trajectory point (rx, ry) to any obstacle."""
        if not obstacles:
            return 4.5
        min_dist = float("inf")
        for obs in obstacles:
            d = math.hypot(rx - obs.x, ry - obs.y) - obs.radius
            if d < min_dist:
                min_dist = d
        return max(0.0, min_dist)

    @staticmethod
    def _wall_distance_at_point(rx: float, ry: float, walls: list) -> float:
        """Distance from trajectory point to the nearest wall (in robot frame)."""
        if not walls:
            return 4.5
        distances = []
        for wall in walls:
            if wall.side == "left":
                distances.append(abs(wall.min_distance - ry))
            elif wall.side == "right":
                distances.append(abs(-wall.min_distance - ry))
            else:
                distances.append(abs(wall.min_distance - rx))
        return min(distances) if distances else 4.5

    @staticmethod
    def _front_clearance(obstacles: list) -> float:
        """Minimum forward clearance among obstacles in front sector."""
        if not obstacles:
            return 4.5
        best = 4.5
        for obs in obstacles:
            if abs(obs.angle) < math.radians(35):
                best = min(best, obs.distance - obs.radius)
        return max(0.0, best)

    def _dt_since_last_cmd(self) -> float:
        if self._last_cmd_time is None:
            return 0.10
        return max(0.02, time.time() - self._last_cmd_time)

    def _update_oscillation_filter(self, w: float, dt: float):
        """EMA of angular change rate for oscillation damping."""
        alpha = 0.30
        raw_change = abs(w - self._prev_w) / max(dt, 0.02)
        self._oscillation_filter = (
            alpha * raw_change + (1.0 - alpha) * self._oscillation_filter
        )

    def get_scoring_summary(self) -> dict:
        """Return a diagnostics dict of the last planning cycle for JSONL logging."""
        candidates = []
        for s in self._last_candidates:
            candidates.append({
                "v": round(s.v, 4),
                "w": round(s.w, 4),
                "cost": round(s.total_cost, 4),
                "clearance": round(s.min_clearance, 4),
                "goal_align": round(s.goal_alignment, 4),
                "safe": s.is_safe,
            })

        summary = {
            "dwa_used": not self._last_was_fallback,
            "candidates": candidates,
        }

        if self._last_selected is not None:
            summary["selected"] = {
                "v": round(self._last_selected.v, 4),
                "w": round(self._last_selected.w, 4),
                "cost": round(self._last_selected.total_cost, 4),
                "clearance": round(self._last_selected.min_clearance, 4),
                "goal_align": round(self._last_selected.goal_alignment, 4),
                "safe": self._last_selected.is_safe,
            }

        if self._last_fallback_score is not None:
            summary["fallback"] = {
                "v": round(self._last_fallback_score.v, 4),
                "w": round(self._last_fallback_score.w, 4),
                "cost": round(self._last_fallback_score.total_cost, 4),
                "clearance": round(self._last_fallback_score.min_clearance, 4),
            }

        return summary

    def reset(self):
        """Reset internal state (call on stage transitions)."""
        self._prev_v = 0.0
        self._prev_w = 0.0
        self._last_cmd_time = None
        self._oscillation_filter = 0.0
        self._frame_idx = 0
        self._last_candidates = []
        self._last_selected = None
        self._last_was_fallback = False
        self._last_fallback_score = None
