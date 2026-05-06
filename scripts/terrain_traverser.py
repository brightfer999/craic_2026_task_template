#!/usr/bin/env python3.8

import json
import math
import os
import time
from typing import List, Optional, Tuple

import numpy as np
import rospy
from geometry_msgs.msg import Twist

from button_presser import ButtonDetector, ButtonPresser
from controller_manager import ControllerManager
from dwa_planner import DWAPlanner, DWAParams
from local_lidar_map import LocalLidarMapper
from localization import Localizer
from utils.math_utils import clamp, normalize_angle
from waypoints import BUTTON_PANELS, StageDirective


class TerrainTraverser:
    """Perception-driven movement primitives for scene 1."""

    _LATERAL_BLEND_SEC = 3.5
    _CREEP_LINEAR = 0.04

    def __init__(self, localizer: Localizer, controller: ControllerManager):
        self._localizer = localizer
        self._controller = controller
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._lidar = LocalLidarMapper()
        self._button_detector = ButtonDetector()
        self._button_presser = ButtonPresser()
        self.obstacle_hit_count = 0

        self._lidar_need_frames = int(rospy.get_param("~lidar_warmup_frames", 8))
        self._lidar_warmup_timeout = float(rospy.get_param("~lidar_warmup_timeout", 4.5))

        # ====== 纯相对坐标策略参数（右墙跟随 + 三重保险体系） ======
        self._WALL_FOLLOW_DISTANCE = float(rospy.get_param("~wall_follow_distance", 0.8))
        self._WALL_FOLLOW_KP_LAT = float(rospy.get_param("~wall_follow_kp_lat", 0.22))
        self._WALL_FOLLOW_KP_ANG = float(rospy.get_param("~wall_follow_kp_ang", 0.20))
        self._CONE_DIAMETER = float(rospy.get_param("~cone_diameter", 0.15))
        self._CONE_MIN_POINTS = int(rospy.get_param("~cone_min_points", 3))
        self._CONE_SAFE_DIST = float(rospy.get_param("~cone_safe_distance", 0.50))
        self._SIDE_SAFE_DIST = float(rospy.get_param("~side_safe_distance", 0.42))
        self._FRONT_STOP_DIST = float(rospy.get_param("~front_stop_distance", 0.36))
        self._FRONT_BACKOFF_DIST = float(rospy.get_param("~front_backoff_distance", 0.22))
        self._CORNER_DETECT_THRESH = float(rospy.get_param("~corner_detect_threshold", 2.5))
        self._CORNER_TURN_W = float(rospy.get_param("~corner_turn_w", 0.35))
        self._LEFT_CORNER_TURN_W = float(rospy.get_param("~left_corner_turn_w", 0.35))
        self._DECAY_FACTOR = float(rospy.get_param("~decay_factor", 0.50))
        self._V_MAX = float(rospy.get_param("~v_max", 0.25))
        self._WALL_SCORE_THRESH = float(rospy.get_param("~wall_score_threshold", 0.50))
        self._SPARSITY_THRESH = float(rospy.get_param("~sparsity_threshold", 0.35))
        self._MAX_TURN_ANGLE = math.radians(float(rospy.get_param("~max_turn_angle_deg", 126.0)))
        self._initial_yaw = None
        self._OBSTACLE_MAX_YAW = math.radians(float(rospy.get_param("~obstacle_max_yaw_deg", 55.0)))
        self._OBSTACLE_YAW_KP = float(rospy.get_param("~obstacle_yaw_kp", 0.70))
        self._GAP_LOOKAHEAD = float(rospy.get_param("~obstacle_gap_lookahead", 2.2))
        self._GAP_SAFETY_RADIUS = float(rospy.get_param("~obstacle_gap_safety_radius", 0.40))
        self._WALL_GUARD_DIST = float(rospy.get_param("~obstacle_wall_guard_distance", 0.36))
        self._PLANNER_ANGLES_DEG = self._load_planner_angles()
        self._PLANNER_MIN_REWARD = float(rospy.get_param("~planner_min_reward", -2.5))
        self._last_avoidance_clearance = self._lidar.max_range
        self._last_planner_score = 0.0
        self._last_planner_angle = 0.0
        self._planner_frame_idx = 0
        self._planner_log_file = None
        self._planner_log_path = rospy.get_param(
            "~planner_log_path",
            os.environ.get("CRAIC_PLANNER_LOG", ""),
        )
        # ====== 主动防撞墙（Phase 4 新增） ======
        self._WALL_GUARD_TRIGGER = float(rospy.get_param("~wall_guard_trigger_dist", 0.55))
        self._WALL_GUARD_TURN_GAIN = float(rospy.get_param("~wall_guard_turn_gain", 0.70))
        self._CORRIDOR_BIAS_OVERRIDE = rospy.get_param("~corridor_bias_startup", None)

        # ====== DWA 局部规划器（Phase 2 规划层替换） ======
        self._dwa_params = DWAParams.from_ros_params()
        self._dwa_planner = DWAPlanner(params=self._dwa_params, lidar_mapper=self._lidar)
        self._use_dwa = rospy.get_param("~use_dwa", True)
        self._open_planner_log()

        # ====== 锥桶看门狗：巡逻中未发现锥桶则掉头 ======
        self._cones_confirmed = False
        self.no_cone_abort = False
        self._cone_abort_distance = float(rospy.get_param("~no_cone_reversal_distance", 3.0))

    def _load_planner_angles(self) -> List[float]:
        raw = rospy.get_param("~planner_candidate_angles_deg", [-55, -40, -28, -16, 0, 16, 28, 40, 55])
        try:
            return [math.radians(float(v)) for v in raw]
        except Exception:
            return [math.radians(v) for v in (-55, -40, -28, -16, 0, 16, 28, 40, 55)]

    def _open_planner_log(self) -> None:
        if not self._planner_log_path:
            return
        try:
            directory = os.path.dirname(os.path.abspath(self._planner_log_path))
            if directory:
                os.makedirs(directory, exist_ok=True)
            self._planner_log_file = open(self._planner_log_path, "a", encoding="utf-8")
            self._write_planner_event(
                "start",
                {
                    "params": {
                        "gap_safety_radius": self._GAP_SAFETY_RADIUS,
                        "gap_lookahead": self._GAP_LOOKAHEAD,
                        "front_stop_distance": self._FRONT_STOP_DIST,
                        "front_backoff_distance": self._FRONT_BACKOFF_DIST,
                        "planner_min_reward": self._PLANNER_MIN_REWARD,
                        "v_max": self._V_MAX,
                        "candidate_angles_deg": [math.degrees(a) for a in self._PLANNER_ANGLES_DEG],
                    },
                    "dwa_params": {
                        "v_max": self._dwa_params.v_max,
                        "w_max": self._dwa_params.w_max,
                        "goal_heading_weight": self._dwa_params.goal_heading_weight,
                        "clearance_reward_gain": self._dwa_params.clearance_reward_gain,
                        "memory_cost_weight": self._dwa_params.memory_cost_weight,
                        "oscillation_weight": self._dwa_params.angular_oscillation_weight,
                        "goal_proximity_reward": self._dwa_params.goal_proximity_reward,
                        "forward_progress_weight": self._dwa_params.forward_progress_weight,
                        "caution_turn_reward": self._dwa_params.caution_turn_reward,
                        "danger_extra_cost": self._dwa_params.danger_extra_cost,
                        "obstacle_danger_dist": self._dwa_params.obstacle_danger_dist,
                        "obstacle_caution_dist": self._dwa_params.obstacle_caution_dist,
                        "min_acceptable_score": self._dwa_params.min_acceptable_score,
                    },
                },
            )
            rospy.loginfo("Planner JSONL logging enabled: %s", self._planner_log_path)
        except Exception as exc:
            self._planner_log_file = None
            rospy.logwarn("Planner log open failed: %s", exc)

    def _write_planner_event(self, event: str, payload: dict) -> None:
        if self._planner_log_file is None:
            return
        record = {
            "event": event,
            "wall_time": time.time(),
        }
        try:
            record["ros_time"] = rospy.Time.now().to_sec()
        except Exception:
            record["ros_time"] = 0.0
        record.update(payload)
        try:
            self._planner_log_file.write(json.dumps(record, sort_keys=True) + "\n")
            self._planner_log_file.flush()
        except Exception as exc:
            rospy.logwarn_throttle(5.0, "Planner log write failed: %s", exc)

    def _log_planner_frame(
        self,
        cmd: Twist,
        pose,
        yaw_delta: float,
        obstacles,
        walls,
        target_angle: float,
        score: float,
        path_clearance: float,
        raw_front: float,
        nearest_clearance: float,
        dwa_summary: dict = None,
        grid_summary: dict = None,
        cone_list: list = None,
        is_escaped: bool = None,
        is_revisiting: bool = None,
    ) -> None:
        if self._planner_log_file is None:
            return
        self._planner_frame_idx += 1
        occ, visited = self._lidar.memory_size()
        obstacle_payload = [
            {
                "x": round(float(obs.x), 4),
                "y": round(float(obs.y), 4),
                "distance": round(float(obs.distance), 4),
                "angle": round(float(obs.angle), 4),
                "confidence": round(float(obs.confidence), 4),
                "points": int(obs.point_count),
                "span": int(obs.sector_span),
            }
            for obs in obstacles[:12]
        ]
        wall_payload = [
            {
                "side": wall.side,
                "score": round(float(wall.score), 4),
                "min_distance": round(float(wall.min_distance), 4),
                "points": int(wall.point_count),
            }
            for wall in walls[:6]
        ]
        cone_payload = []
        if cone_list:
            for cone in cone_list[:12]:
                cone_payload.append({
                    "x": round(float(cone.x), 4),
                    "y": round(float(cone.y), 4),
                    "distance": round(float(cone.distance), 4),
                    "radius": round(float(cone.radius), 4),
                    "side": cone.side,
                    "points": int(cone.point_count),
                })
        nearest_obstacle = nearest_clearance
        for obs in obstacles:
            nearest_obstacle = min(nearest_obstacle, obs.distance - obs.radius)
        if grid_summary is None:
            grid_summary = self._compute_grid_summary()
        record = {
            "frame": self._planner_frame_idx,
            "stage_distance": self._localizer.distance_since_stage_reset(),
            "pose": {
                "x": float(getattr(pose, "x", 0.0)),
                "y": float(getattr(pose, "y", 0.0)),
                "yaw": float(getattr(pose, "yaw", 0.0)),
                "valid": bool(getattr(pose, "valid", False)),
            },
            "yaw_delta": float(yaw_delta),
            "selected_angle": float(target_angle),
            "score": float(score),
            "path_clearance": float(path_clearance),
            "raw_front": float(raw_front),
            "nearest_clearance": float(nearest_clearance),
            "nearest_obstacle": round(float(nearest_obstacle), 4),
            "obstacle_count": len(obstacles),
            "wall_count": len(walls),
            "cone_count": len(cone_list) if cone_list else 0,
            "memory_occupied": occ,
            "memory_visited": visited,
            "cmd": {
                "linear_x": float(cmd.linear.x),
                "linear_y": float(cmd.linear.y),
                "angular_z": float(cmd.angular.z),
            },
            "obstacles": obstacle_payload,
            "walls": wall_payload,
            "cones": cone_payload,
        }
        if dwa_summary is not None:
            record["dwa"] = dwa_summary
        if grid_summary is not None:
            record["grid_map"] = grid_summary
        if is_escaped is not None:
            record["is_escaped"] = is_escaped
        if is_revisiting is not None:
            record["is_revisiting"] = is_revisiting
        self._write_planner_event("frame", record)

    def wait_for_lidar(self, timeout: float = 6.0) -> bool:
        """Wait until the local lidar mapper has received at least one cloud."""
        deadline = time.time() + timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._lidar.has_data:
                return True
            rate.sleep()
        return self._lidar.has_data

    def log_obstacle_status(self, prefix: str = "obstacle") -> None:
        """Log a compact local-perception snapshot for seed0 debugging."""
        clearance = self._lidar.clearance()
        raw = self._lidar.raw_clearance()
        walls = self._lidar.detect_walls()
        cones = self._lidar.detect_cones_with_exclusion(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
        )
        gap = self._lidar.find_best_gap(
            lookahead=self._GAP_LOOKAHEAD,
            safety_radius=self._GAP_SAFETY_RADIUS,
        )

        if not clearance.has_data:
            rospy.logwarn("[%s] lidar has no data", prefix)
            return

        rospy.loginfo(
            "[%s] clear f=%.2f lf=%.2f rf=%.2f l=%.2f r=%.2f | "
            "raw_front=%.2f | cones=%d walls=%s | gap=%s angle=%.1f width=%.1f",
            prefix,
            clearance.front,
            clearance.left_front,
            clearance.right_front,
            clearance.left,
            clearance.right,
            min(raw.front, raw.left_front, raw.right_front) if raw.has_data else self._lidar.max_range,
            len(cones),
            ",".join("%s:%.2f" % (w.side, w.score) for w in walls[:3]) or "none",
            "ok" if gap.has_gap else "blocked",
            math.degrees(gap.angle),
            math.degrees(gap.width),
        )
        occ, visited = self._lidar.memory_size()
        rospy.loginfo(
            "[%s] planner score=%.2f angle=%.1f last_clear=%.2f memory occ=%d visited=%d",
            prefix,
            self._last_planner_score,
            math.degrees(self._last_planner_angle),
            self._last_avoidance_clearance,
            occ,
            visited,
        )

    def perform_startup_lidar_rotation(
        self, duration: Optional[float] = None, angular_z: Optional[float] = None
    ) -> bool:
        """Slow in-place yaw so successive lidar clouds sample the surroundings."""
        if duration is None:
            duration = float(rospy.get_param("~startup_lidar_spin_duration", 10.0))
        if angular_z is None:
            angular_z = float(rospy.get_param("~startup_lidar_spin_speed", -0.34))

        rospy.loginfo("Startup lidar scan: %.1fs at wz=%.2f rad/s", duration, angular_z)
        rate = rospy.Rate(20)
        deadline = time.time() + duration
        cmd = Twist()
        cmd.angular.z = clamp(angular_z, -0.5, 0.0)

        _last_watchdog = time.time()
        while not rospy.is_shutdown() and time.time() < deadline:
            self._cmd_pub.publish(cmd)
            rate.sleep()
            if time.time() - _last_watchdog >= 0.8:
                if self._check_cones_watchdog():
                    self.stop(0.3)
                    return False
                _last_watchdog = time.time()

        self.stop()
        return True

    def cross_seed0_obstacle_zone(
        self,
        leave_stage: StageDirective,
        obstacle_stage: StageDirective,
    ) -> bool:
        """Run the seed0 acceptance path without depending on the full scene FSM."""
        self.reset_cone_tracking()
        leave_ok = self.follow_corridor(leave_stage, reverse=False)
        self.log_obstacle_status(prefix="after_leave_start")
        obstacle_ok = self.cross_obstacle_zone(obstacle_stage, reverse=False)
        self.log_obstacle_status(prefix="after_obstacle")
        return leave_ok and obstacle_ok

    def explore_and_detect_obstacles(self) -> None:
        rospy.loginfo("=== 探索并检测障碍物 ===")

        clearance = self._lidar.clearance()
        if not clearance.has_data:
            rospy.logwarn("No lidar data for obstacle exploration")
            return

        rospy.loginfo(
            "Clearance: front=%.2f lf=%.2f rf=%.2f left=%.2f right=%.2f",
            clearance.front, clearance.left_front, clearance.right_front,
            clearance.left, clearance.right,
        )

        dist_l, num_l, _ = self._lidar.left_cone_info(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
            sparsity_threshold=self._SPARSITY_THRESH,
        )
        dist_r, num_r, _ = self._lidar.right_cone_info(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
            sparsity_threshold=self._SPARSITY_THRESH,
        )

        if num_l > 0 or num_r > 0:
            self._cones_confirmed = True
            rospy.loginfo("Cones found: L=%d(%.2fm) R=%d(%.2fm)", num_l, dist_l, num_r, dist_r)
        else:
            rospy.loginfo("No cones in immediate surroundings")

        if clearance.front < 0.5:
            rospy.logwarn("Nearby obstacle in front (%.2fm)", clearance.front)
            self.obstacle_hit_count += 1

    def run_stage(self, stage: StageDirective, reverse: bool = False) -> bool:
        """Run a stage directive by dispatching to the appropriate method."""
        self.reset_cone_tracking()
        if stage.name == "obstacle":
            return self.cross_obstacle_zone(stage, reverse=reverse)
        elif stage.name in ("operator_1", "operator_2"):
            return self.press_panel(stage.name)
        elif stage.name == "terrain":
            return self.follow_corridor(stage, reverse=reverse)
        else:
            return self.follow_corridor(stage, reverse=reverse)

    def cross_obstacle_zone(self, stage: StageDirective, reverse: bool = False) -> bool:
        self._localizer.reset_stage_progress()
        self._initial_yaw = self._localizer.yaw_since_stage_reset()
        self.no_cone_abort = False
        self._dwa_planner.reset()
        start = time.time()
        direction = 1.0
        if reverse:
            rospy.logwarn("Obstacle zone reverse requested; using forward-locked gap following")
        self._run_lidar_warmup_gate(direction)
        rate = rospy.Rate(20)
        _last_log = time.time()
        _dwa_fallback_count = 0

        while not rospy.is_shutdown() and time.time() - start < stage.timeout:
            travelled = self._localizer.distance_since_stage_reset()
            if travelled >= stage.nominal_distance:
                self.stop(0.5)
                return True

            clearance = self._lidar.clearance()
            if clearance.has_data and clearance.front < 0.42:
                self.obstacle_hit_count += 1

            # --- Perception snapshot for both planners ---
            raw = self._lidar.raw_clearance()
            raw_front = self._lidar.max_range
            if raw.has_data:
                raw_front = min(raw.front, raw.left_front, raw.right_front)

            pose = self._localizer.get_pose()
            if pose.valid:
                self._lidar.update_local_memory(pose.x, pose.y, pose.yaw)

            if self._initial_yaw is None:
                self._initial_yaw = self._localizer.yaw_since_stage_reset()
            current_yaw = self._localizer.yaw_since_stage_reset()
            yaw_delta = normalize_angle(current_yaw - self._initial_yaw)

            obstacles = self._lidar.detect_obstacles_cross_sector(
                max_distance=2.6,
                cluster_radius=max(0.20, self._CONE_DIAMETER * 1.7),
                min_points=self._CONE_MIN_POINTS,
            )
            walls = self._lidar.detect_walls()
            cones = self._lidar.detect_cones_with_exclusion(
                cone_diameter=self._CONE_DIAMETER,
                cone_min_points=self._CONE_MIN_POINTS,
            )
            if cones:
                self._cones_confirmed = True

            # --- DWA planning (primary) ---
            cmd = Twist()
            used_fallback = True

            if self._use_dwa and getattr(stage, "use_dwa", False) and clearance.has_data:
                dwa_cmd, dwa_score, fallback = self._dwa_planner.plan(
                    clearance=clearance,
                    obstacles=obstacles,
                    walls=walls,
                    pose=pose,
                    corridor_bias=stage.corridor_bias,
                    yaw_delta=yaw_delta,
                    preferred_direction=direction,
                )
                if not fallback:
                    cmd = dwa_cmd
                    used_fallback = False
                    # Phase 4: log DWA planning frame
                    dwa_summary = self._dwa_planner.get_scoring_summary()
                    grid_summary = self._compute_grid_summary()
                    nearest_clearance = min(raw_front, *(obs.distance - obs.radius for obs in obstacles)) if obstacles else raw_front
                    self._log_planner_frame(
                        cmd=cmd,
                        pose=pose,
                        yaw_delta=yaw_delta,
                        obstacles=obstacles,
                        walls=walls,
                        target_angle=float(dwa_score.w),
                        score=float(-dwa_score.total_cost),
                        path_clearance=float(dwa_score.min_clearance),
                        raw_front=raw_front,
                        nearest_clearance=float(nearest_clearance),
                        dwa_summary=dwa_summary,
                        grid_summary=grid_summary,
                        cone_list=cones,
                        is_escaped=None,
                        is_revisiting=None,
                    )
                else:
                    cmd = dwa_cmd
                    used_fallback = False
                    _dwa_fallback_count += 1
                    self._log_planner_frame(
                        cmd=cmd,
                        pose=pose,
                        yaw_delta=yaw_delta,
                        obstacles=obstacles,
                        walls=walls,
                        target_angle=float(dwa_score.w),
                        score=float(-dwa_score.total_cost),
                        path_clearance=float(dwa_score.min_clearance),
                        raw_front=raw_front,
                        nearest_clearance=float(raw_front),
                        dwa_summary=self._dwa_planner.get_scoring_summary(),
                        grid_summary=self._compute_grid_summary(),
                        cone_list=cones,
                        is_escaped=True,
                        is_revisiting=None,
                    )

            if used_fallback:
                # Existing gap-follow planner as fallback (logs internally)
                cmd = self._cone_zone_gap_follow_cmd(
                    stage.speed,
                    lateral_scale=1.0,
                    angular_scale=1.0,
                )

            self._cmd_pub.publish(cmd)
            rate.sleep()

            if time.time() - _last_log >= 1.0:
                self._log_obstacle_perception()
                if self._use_dwa:
                    rospy.loginfo(
                        "DWA status: fallback_rate=%.1f%%",
                        100.0 * _dwa_fallback_count / max(1, self._dwa_planner._frame_idx),
                    )
                _last_log = time.time()

        self.stop()
        rospy.logwarn(
            "Obstacle stage timed out after %.1fm",
            self._localizer.distance_since_stage_reset(),
        )
        return self._localizer.distance_since_stage_reset() >= 0.65 * stage.nominal_distance

    def follow_corridor(self, stage: StageDirective, reverse: bool = False) -> bool:
        return self.advance_until_tag_or_distance(stage, reverse=reverse, stop_on_tag=False)

    def advance_until_tag_or_distance(
        self, stage: StageDirective, reverse: bool = False, stop_on_tag: bool = False
    ) -> bool:
        self._localizer.reset_stage_progress()
        self._initial_yaw = None
        start = time.time()
        direction = -1.0 if reverse else 1.0
        rate = rospy.Rate(20)
        _last_watchdog = time.time()

        while not rospy.is_shutdown() and time.time() - start < stage.timeout:
            travelled = self._localizer.distance_since_stage_reset()
            if travelled >= stage.nominal_distance:
                self.stop(0.5)
                return True

            if stop_on_tag and stage.target_tags:
                if self._localizer.has_seen_tag(stage.target_tags):
                    self.stop(0.5)
                    rospy.loginfo("Tag %s detected, stopping", stage.target_tags)
                    return True

            corridor_bias = stage.corridor_bias
            if self._CORRIDOR_BIAS_OVERRIDE is not None:
                corridor_bias = float(self._CORRIDOR_BIAS_OVERRIDE)
            cmd = self._reactive_cmd(
                stage.speed, direction, corridor_bias,
                lateral_scale=1.0, angular_scale=1.0, reverse=reverse,
            )
            self._cmd_pub.publish(cmd)
            rate.sleep()

            if time.time() - _last_watchdog >= 0.8:
                if self._check_cones_watchdog():
                    self.stop(0.3)
                    return False
                _last_watchdog = time.time()

        self.stop()
        travelled = self._localizer.distance_since_stage_reset()
        rospy.logwarn(
            "advance_until_tag_or_distance timed out after %.1fm", travelled,
        )
        return travelled >= 0.65 * stage.nominal_distance

    def press_panel(self, panel_name: str) -> bool:
        panel = BUTTON_PANELS[panel_name]
        self._controller.use_mpc()
        approach_stage = StageDirective(
            name=f"{panel_name}_approach",
            controller="mpc",
            nominal_distance=panel["approach_distance"],
            timeout=panel["approach_timeout"],
            speed=0.08,
        )
        self.advance_until_tag_or_distance(approach_stage)

        color = self._detect_lit_button()
        if color is None:
            rospy.logwarn("%s: no lit button detected, falling back to red", panel_name)
            color = "red"

        rospy.loginfo("%s: pressing %s button", panel_name, color)
        self._button_presser.press_button_with_arm(arm=panel["arm"], press_duration=0.8)
        rospy.sleep(1.0)

        still_lit = self._button_detector.is_lit(color)
        if still_lit is True:
            rospy.logwarn("%s: %s button still appears lit after press", panel_name, color)
            return False

        self._back_off(0.35)
        return True

    def stop(self, duration: float = 0.2):
        cmd = Twist()
        rate = rospy.Rate(20)
        end_time = time.time() + duration
        while not rospy.is_shutdown() and time.time() < end_time:
            self._cmd_pub.publish(cmd)
            rate.sleep()

    def _check_cones_watchdog(self) -> bool:
        """Dual-side cone watchdog: aborts stage if no cones after 3m."""
        if self._cones_confirmed:
            return False

        dist_l, num_l, _ = self._lidar.left_cone_info(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
            sparsity_threshold=self._SPARSITY_THRESH,
        )
        dist_r, num_r, _ = self._lidar.right_cone_info(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
            sparsity_threshold=self._SPARSITY_THRESH,
        )

        if num_l > 0 or num_r > 0:
            side = "L" if num_l > num_r else "R"
            self._cones_confirmed = True
            rospy.loginfo("Watchdog: cones on %s L=%d R=%d min=%.2fm",
                          side, num_l, num_r, min(dist_l, dist_r))
            return False

        distance = self._localizer.distance_since_stage_reset()
        if distance > self._cone_abort_distance:
            rospy.logwarn("Watchdog: %.2fm no cones, trigger reversal", distance)
            self.no_cone_abort = True
            return True

        return False

    def perform_u_turn(self):
        """Safe 180-degree turn with lidar obstacle avoidance."""
        rospy.loginfo("=== Safe 180 U-turn ===")
        self.stop(0.3)

        pose = self._localizer.get_pose()
        start_yaw = pose.yaw
        target_yaw = normalize_angle(start_yaw + math.pi)

        rate = rospy.Rate(20)
        deadline = time.time() + 18.0
        base_speed = 0.35

        while not rospy.is_shutdown() and time.time() < deadline:
            pose = self._localizer.get_pose()
            yaw_diff = normalize_angle(target_yaw - pose.yaw)

            if abs(yaw_diff) < 0.08:
                break

            c = self._lidar.clearance()
            raw = self._lidar.raw_clearance()

            if raw.has_data:
                raw_min = min(raw.front, raw.right_front, raw.left_front,
                              raw.right * 0.5)
            else:
                raw_min = 99.0

            cmd = Twist()

            if raw_min < 0.25:
                cmd.linear.x = -0.06
                cmd.angular.z = 0.0
                self._cmd_pub.publish(cmd)
                rospy.sleep(0.4)
                continue

            near = 99.0
            if c.has_data:
                near = min(c.front, c.right_front, c.left_front)

            speed = base_speed * 0.4 if near < 0.55 else base_speed
            cmd.angular.z = math.copysign(speed, yaw_diff)

            if c.has_data and c.left_front < 0.40 and cmd.angular.z > 0:
                cmd.angular.z = -0.20
            if c.has_data and c.right_front < 0.40 and cmd.angular.z < 0:
                cmd.angular.z = 0.20

            self._cmd_pub.publish(cmd)
            rate.sleep()

        self.stop(0.3)
        self.no_cone_abort = False
        self._cones_confirmed = False
        self._initial_yaw = None
        rospy.loginfo("U-turn done, state reset")

    def reset_cone_tracking(self):
        self._cones_confirmed = False
        self.no_cone_abort = False
        self._initial_yaw = None

    def _check_corner(self, clearance) -> bool:
        """检测右拐角：左侧墙消失（左前+左侧扇区均开阔）。"""
        if not clearance.has_data:
            return False
        return (
            clearance.left > self._CORNER_DETECT_THRESH
            and clearance.left_front > self._CORNER_DETECT_THRESH * 0.75
        )

    def _check_left_corner(self, clearance) -> bool:
        """检测左拐角：右侧墙消失（右前+右侧扇区均开阔）。"""
        if not clearance.has_data:
            return False
        return (
            clearance.right > self._CORNER_DETECT_THRESH
            and clearance.right_front > self._CORNER_DETECT_THRESH * 0.75
        )

    def _lateral_angular_blend(self, elapsed: float, blend_motion: bool) -> Tuple[float, float]:
        if not blend_motion:
            return 1.0, 1.0
        if elapsed >= self._LATERAL_BLEND_SEC:
            return 1.0, 1.0
        t = elapsed / self._LATERAL_BLEND_SEC
        scale = t * t * (3.0 - 2.0 * t)
        return scale, scale

    def _run_lidar_warmup_gate(self, direction: float) -> None:
        deadline = time.time() + self._lidar_warmup_timeout
        rate = rospy.Rate(10)
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._lidar.has_data:
                rospy.loginfo("Lidar warmup gate passed (direction=%.1f)", direction)
                return
            rate.sleep()
        rospy.logwarn("Lidar warmup gate timed out after %.1fs", self._lidar_warmup_timeout)

    def _cone_zone_gap_follow_cmd(
        self,
        speed: float,
        lateral_scale: float = 1.0,
        angular_scale: float = 1.0,
    ) -> Twist:
        """Forward-locked obstacle navigation using reward-scored local planning."""
        cmd = Twist()
        if not self._lidar.has_data:
            cmd.linear.x = min(speed, 0.08)
            return cmd

        if self._initial_yaw is None:
            self._initial_yaw = self._localizer.yaw_since_stage_reset()

        current_yaw = self._localizer.yaw_since_stage_reset()
        yaw_delta = normalize_angle(current_yaw - self._initial_yaw)
        preferred_angle = clamp(-yaw_delta, -math.radians(35), math.radians(35))

        raw = self._lidar.raw_clearance()
        raw_front = self._lidar.max_range
        left_space = self._lidar.max_range
        right_space = self._lidar.max_range
        if raw.has_data:
            raw_front = min(raw.front, raw.left_front, raw.right_front)
            left_space = min(raw.left, raw.left_front)
            right_space = min(raw.right, raw.right_front)

        pose = self._localizer.get_pose()
        if pose.valid:
            self._lidar.update_local_memory(pose.x, pose.y, pose.yaw)

        walls = self._lidar.detect_walls()
        cones = self._lidar.detect_cones_with_exclusion(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
        )
        obstacles = self._lidar.detect_obstacles_cross_sector(
            max_distance=max(2.2, self._GAP_LOOKAHEAD + 0.4),
            cluster_radius=max(0.20, self._CONE_DIAMETER * 1.7),
            min_points=self._CONE_MIN_POINTS,
        )
        if cones:
            self._cones_confirmed = True

        front_wall_distance = self._lidar.max_range
        for wall in walls:
            if wall.side == "front":
                front_wall_distance = min(front_wall_distance, wall.min_distance)

        nearest_obstacle = min([obs.distance for obs in obstacles] or [self._lidar.max_range])
        nearest_clearance = min(raw_front, nearest_obstacle, front_wall_distance)
        self._last_avoidance_clearance = nearest_clearance

        if nearest_clearance < self._FRONT_BACKOFF_DIST:
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            safe_turn = -1.0 if right_space >= left_space else 1.0
            cmd.angular.z = clamp(0.22 * safe_turn - self._OBSTACLE_YAW_KP * yaw_delta, -0.32, 0.32)
            self._last_planner_score = -9.0
            self._last_planner_angle = 0.0
            self._log_planner_frame(
                cmd=cmd,
                pose=pose,
                yaw_delta=yaw_delta,
                obstacles=obstacles,
                walls=walls,
                target_angle=0.0,
                score=-9.0,
                path_clearance=nearest_clearance,
                raw_front=raw_front,
                nearest_clearance=nearest_clearance,
            )
            rospy.logwarn_throttle(0.8, "Planner emergency stop %.2fm", nearest_clearance)
            return cmd

        yaw_locked = abs(yaw_delta) > self._OBSTACLE_MAX_YAW
        if yaw_locked:
            cmd.linear.x = 0.05 if nearest_clearance > self._FRONT_STOP_DIST else 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = clamp(-self._OBSTACLE_YAW_KP * yaw_delta, -0.30, 0.30) * angular_scale
            self._last_planner_score = -5.0
            self._last_planner_angle = 0.0
            self._log_planner_frame(
                cmd=cmd,
                pose=pose,
                yaw_delta=yaw_delta,
                obstacles=obstacles,
                walls=walls,
                target_angle=0.0,
                score=-5.0,
                path_clearance=nearest_clearance,
                raw_front=raw_front,
                nearest_clearance=nearest_clearance,
            )
            rospy.logwarn_throttle(
                0.8,
                "Obstacle yaw recovery %.1f deg; keep forward direction",
                math.degrees(yaw_delta),
            )
            return cmd

        target_angle, score, path_clearance = self._score_obstacle_candidates(
            preferred_angle=preferred_angle,
            obstacles=obstacles,
            walls=walls,
            pose=pose,
            yaw_delta=yaw_delta,
        )
        self._last_planner_score = score
        self._last_planner_angle = target_angle
        self._last_avoidance_clearance = min(self._last_avoidance_clearance, path_clearance)

        if path_clearance < 0.30:
            vx = min(speed, 0.035)
        elif path_clearance < 0.40:
            vx = min(speed, 0.075)
        elif path_clearance < 0.75:
            vx = min(speed, 0.12)
        else:
            vx = min(speed * 1.18, min(self._V_MAX, 0.20))

        if score < self._PLANNER_MIN_REWARD:
            vx = min(vx, 0.045)

        lateral = clamp(0.32 * math.sin(target_angle), -0.18, 0.18)
        w = 0.95 * target_angle - self._OBSTACLE_YAW_KP * yaw_delta

        cmd.linear.x = clamp(vx, 0.0, min(self._V_MAX, 0.20))
        cmd.linear.y = clamp(lateral, -0.18, 0.18) * lateral_scale
        cmd.angular.z = clamp(w, -0.35, 0.35) * angular_scale
        self._log_planner_frame(
            cmd=cmd,
            pose=pose,
            yaw_delta=yaw_delta,
            obstacles=obstacles,
            walls=walls,
            target_angle=target_angle,
            score=score,
            path_clearance=path_clearance,
            raw_front=raw_front,
            nearest_clearance=nearest_clearance,
        )
        rospy.loginfo_throttle(
            0.8,
            "Planner target=%.1f score=%.2f clear=%.2f obs=%d",
            math.degrees(target_angle),
            score,
            path_clearance,
            len(obstacles),
        )
        return cmd

    def _score_obstacle_candidates(self, preferred_angle, obstacles, walls, pose, yaw_delta):
        best_angle = preferred_angle
        best_score = -float("inf")
        best_clearance = self._lidar.max_range
        for angle in self._PLANNER_ANGLES_DEG:
            clearance, collision_loss = self._candidate_clearance_and_loss(angle, obstacles)
            reward = 0.0
            loss = 0.0

            reward += 2.4 * max(0.0, math.cos(angle))
            reward += 1.3 * clamp((clearance - 0.30) / 1.0, 0.0, 1.0)
            reward += 0.8 * clamp(1.0 - abs(angle - preferred_angle) / math.radians(75), 0.0, 1.0)

            if 0.30 <= clearance <= 0.40 and abs(angle) > math.radians(10):
                reward += 0.9
            if self._last_avoidance_clearance < 0.30:
                loss += 2.2
            elif 0.30 <= self._last_avoidance_clearance <= 0.40 and abs(angle) > math.radians(10):
                reward += 0.5

            loss += collision_loss
            loss += 0.9 * abs(angle) / math.radians(75)
            loss += 0.6 * abs(normalize_angle(yaw_delta + angle))
            loss += self._candidate_memory_loss(angle, pose)
            loss += self._candidate_wall_loss(angle, walls)

            if math.cos(angle) < 0.45:
                loss += 3.0

            score = reward - loss
            if score > best_score:
                best_score = score
                best_angle = angle
                best_clearance = clearance

        return clamp(best_angle, -math.radians(55), math.radians(55)), best_score, best_clearance

    def _candidate_clearance_and_loss(self, angle: float, obstacles) -> Tuple[float, float]:
        ca = math.cos(angle)
        sa = math.sin(angle)
        clearance = self._lidar.max_range
        loss = 0.0
        for obs in obstacles:
            forward = obs.x * ca + obs.y * sa
            lateral = abs(-obs.x * sa + obs.y * ca)
            if forward <= 0.0 or forward > self._GAP_LOOKAHEAD:
                continue
            guard = self._GAP_SAFETY_RADIUS + obs.radius
            if lateral < guard:
                local_clear = max(0.0, forward - obs.radius)
                clearance = min(clearance, local_clear)
                risk = (guard - lateral) / max(0.05, guard)
                loss += (1.2 + 2.5 * risk) * obs.confidence
            elif lateral < guard + 0.25:
                loss += 0.45 * obs.confidence
        return clearance, loss

    def _candidate_memory_loss(self, angle: float, pose) -> float:
        if not getattr(pose, "valid", False):
            return 0.0
        cy = math.cos(pose.yaw)
        sy = math.sin(pose.yaw)
        loss = 0.0
        for dist in (0.35, 0.70, 1.05, 1.40, 1.75):
            lx = dist * math.cos(angle)
            ly = dist * math.sin(angle)
            gx = pose.x + cy * lx - sy * ly
            gy = pose.y + sy * lx + cy * ly
            loss += self._lidar.memory_cost_at(gx, gy)
        return 0.35 * loss

    def _candidate_wall_loss(self, angle: float, walls) -> float:
        loss = 0.0
        for wall in walls:
            if wall.side == "front" and abs(angle) < math.radians(18):
                loss += 1.6 * wall.score * clamp((0.75 - wall.min_distance) / 0.75, 0.0, 1.0)
            elif wall.side == "left" and angle > 0:
                loss += 0.8 * wall.score * clamp((self._WALL_GUARD_DIST - wall.min_distance) / self._WALL_GUARD_DIST, 0.0, 1.0)
            elif wall.side == "right" and angle < 0:
                loss += 0.8 * wall.score * clamp((self._WALL_GUARD_DIST - wall.min_distance) / self._WALL_GUARD_DIST, 0.0, 1.0)
        return loss

    def _compute_grid_summary(self) -> dict:
        """Phase 4: compute a compact grid-map summary for JSONL logging."""
        try:
            grid = self._lidar.build_grid_map()
        except Exception:
            return None
        if grid is None:
            return None
        g = grid.grid
        occupied = int(np.sum(g == 2))  # CellState.OCCUPIED
        inflated = int(np.sum(g == 3))  # CellState.INFLATED
        free = int(np.sum(g == 1))      # CellState.FREE
        unknown = int(np.sum(g == 0))   # CellState.UNKNOWN
        total = max(1, g.size)
        front_region = grid.get_front_region(1.5)
        front_blocked = int(np.sum(front_region == 2) + np.sum(front_region == 3))
        front_total = max(1, front_region.size)
        return {
            "total_cells": int(total),
            "occupied": occupied,
            "inflated": inflated,
            "free": free,
            "unknown": unknown,
            "occupancy_pct": round(100.0 * (occupied + inflated) / total, 2),
            "front_blocked_pct": round(100.0 * front_blocked / front_total, 2),
            "front_clear": grid.front_is_clear(1.2),
        }

    def _log_obstacle_perception(self) -> None:
        # 使用互斥分类：先检测墙壁，再检测锥桶（排除墙壁区域）
        walls = self._lidar.detect_walls()
        cones = self._lidar.detect_cones_with_exclusion(
            cone_diameter=self._CONE_DIAMETER,
            cone_min_points=self._CONE_MIN_POINTS,
        )
        gap = self._lidar.find_best_gap(
            lookahead=self._GAP_LOOKAHEAD,
            safety_radius=self._GAP_SAFETY_RADIUS,
        )
        yaw = 0.0
        if self._initial_yaw is not None:
            yaw = normalize_angle(self._localizer.yaw_since_stage_reset() - self._initial_yaw)
        rospy.loginfo(
            "Obstacle lidar: cones=%d walls=%s gap=%s angle=%.1f yaw=%.1f",
            len(cones),
            ",".join("%s:%.2f" % (w.side, w.score) for w in walls[:3]) or "none",
            "ok" if gap.has_gap else "blocked",
            math.degrees(gap.angle),
            math.degrees(yaw),
        )

    def _reactive_cmd(
        self,
        speed: float,
        direction: float,
        corridor_bias: float = 0.0,
        lateral_scale: float = 1.0,
        angular_scale: float = 1.0,
        reverse: bool = False,
    ) -> Twist:
        """纯相对坐标右墙跟随 + 左锥桶避让 + 双拐角检测。

        策略：紧贴右墙行进，维持 wall_follow_distance 距离。
        三重保险：
          1. v ≥ 0  —— 永不后退
          2. 累计转弯 ≤ max_turn_angle —— 杜绝掉头
          3. 逐帧 w ∈ [-0.50, 0.50]
        """
        if reverse:
            return self._reactive_cmd_reverse(
                speed, direction, corridor_bias, lateral_scale, angular_scale
            )

        clearance = self._lidar.clearance()
        has = clearance.has_data

        right_wall = clearance.right if has else self._lidar.max_range
        front_d = clearance.front if has else self._lidar.max_range
        left_close = min(clearance.left_front, clearance.left) if has else self._lidar.max_range
        right_close = min(clearance.right_front, clearance.right) if has else self._lidar.max_range

        # ========== 1. 右墙跟随 ==========
        wall_err = right_wall - self._WALL_FOLLOW_DISTANCE
        ly = -clamp(self._WALL_FOLLOW_KP_LAT * wall_err, -0.18, 0.18) * lateral_scale
        w_wall = clamp(-self._WALL_FOLLOW_KP_ANG * wall_err, -0.22, 0.35)

        # ========== 2. 侧向锥桶/障碍避让 ==========
        # 侧向锥桶不应触发原地转圈；保持低速前进，同时向安全侧偏航和侧移。
        if left_close < self._CONE_SAFE_DIST:
            avoid = clamp((self._CONE_SAFE_DIST - left_close) / self._CONE_SAFE_DIST, 0.0, 1.0)
            ly = clamp(ly - 0.12 * avoid * lateral_scale, -0.22, 0.22)
            w_wall = clamp(w_wall - 0.28 * avoid, -0.45, 0.35)
        if right_close < self._SIDE_SAFE_DIST:
            avoid = clamp((self._SIDE_SAFE_DIST - right_close) / self._SIDE_SAFE_DIST, 0.0, 1.0)
            ly = clamp(ly + 0.10 * avoid * lateral_scale, -0.22, 0.22)
            w_wall = clamp(w_wall + 0.20 * avoid, -0.45, 0.45)

        # ========== 3. 双级避障（raw clearance 无 EMA 延迟）==========
        raw_c = self._lidar.raw_clearance()
        if raw_c.has_data:
            raw_front_min = min(raw_c.front, raw_c.right_front, raw_c.left_front)
            raw_side_min = min(raw_c.right, raw_c.left)
            right_space = min(raw_c.right_front, raw_c.right)
            left_space = min(raw_c.left_front, raw_c.left)
        else:
            raw_front_min = 99.0
            raw_side_min = 99.0
            right_space = 99.0
            left_space = 99.0

        # 提前初始化累计转弯（紧急避障也需受锁限制）
        if self._initial_yaw is None:
            self._initial_yaw = self._localizer.yaw_since_stage_reset()

        turn_locked = False
        current_yaw = self._localizer.yaw_since_stage_reset()
        yaw_delta = normalize_angle(current_yaw - self._initial_yaw)
        if abs(yaw_delta) > self._MAX_TURN_ANGLE:
            turn_locked = True
            rospy.logwarn_throttle(0.5, "累计转弯超过%.1f°，锁定方向",
                                   math.degrees(self._MAX_TURN_ANGLE))

        # ========== 3.5 主动防撞墙（Phase 4 新增）==========
        # 当任何方向的距离低于 trigger 值时，强制转向远离障碍的方向。
        # 优先级高于右墙跟随，防止"主动靠墙"行为。
        wall_guard_w = 0.0
        wall_guard_active = False
        if raw_front_min < self._WALL_GUARD_TRIGGER or min(left_close, right_close) < self._WALL_GUARD_TRIGGER * 0.65:
            wall_guard_active = True
            if right_space >= left_space + 0.03:
                wall_guard_w = -self._WALL_GUARD_TURN_GAIN  # turn away from left obstacle
            elif left_space >= right_space + 0.03:
                wall_guard_w = self._WALL_GUARD_TURN_GAIN   # turn away from right obstacle
            else:
                # Both sides tight — steer toward the more open side
                wall_guard_w = clamp(
                    (right_space - left_space) * 1.8,
                    -self._WALL_GUARD_TURN_GAIN,
                    self._WALL_GUARD_TURN_GAIN,
                )
            rospy.loginfo_throttle(
                0.5,
                "WallGuard front=%.2f L=%.2f R=%.2f w=%.2f",
                raw_front_min, left_space, right_space, wall_guard_w,
            )

        # ========== 4. 双拐角检测 ==========
        # 只有前方也受限时才按拐角处理，避免把锥桶间隙/侧向开阔误判成转弯口。
        right_corner = self._check_corner(clearance)
        left_corner = self._check_left_corner(clearance)
        corner_allowed = min(front_d, raw_front_min) < 1.05

        if right_corner and corner_allowed:
            w = self._CORNER_TURN_W
            front_d = min(front_d, 0.45)
            rospy.loginfo_throttle(1.0, "检测到右拐角，执行左转弯(避开场景二) w=%.2f", w)
        elif left_corner and corner_allowed:
            w = -self._LEFT_CORNER_TURN_W
            front_d = min(front_d, 0.45)
            rospy.loginfo_throttle(1.0, "检测到左拐角(右侧开阔)，执行右转弯 w=%.2f", w)
        else:
            w = w_wall

        # Wall guard overrides normal steering when too close to any wall
        if wall_guard_active:
            w = wall_guard_w
            vx_max = 0.06  # slow down during wall guard
        else:
            vx_max = self._V_MAX

        if raw_front_min < self._FRONT_BACKOFF_DIST:
            cmd = Twist()
            cmd.linear.x = 0.0 if turn_locked else -0.06
            cmd.linear.y = 0.0
            if not turn_locked:
                cmd.angular.z = -0.50 if right_space >= left_space + 0.05 else 0.50
            rospy.loginfo_throttle(1.0, "极限避障(%.2fm) %s",
                                   raw_front_min, "已锁定" if turn_locked else "转向安全侧")
            return cmd

        if raw_front_min < self._FRONT_STOP_DIST:
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            if not turn_locked:
                cmd.angular.z = -0.50 if right_space >= left_space + 0.05 else 0.50
            rospy.loginfo_throttle(1.0, "避障刹车(%.2fm) %s",
                                   raw_front_min, "已锁定" if turn_locked else "转向安全侧")
            return cmd

        # 用 raw 值加速速度衰减（EMA 有延迟）
        front_d = min(front_d, raw_front_min)

        # ========== 5. 速度衰减（前方越近越慢）==========
        if front_d < 0.45:
            sp_factor = 0.12
        elif front_d < 0.75:
            sp_factor = self._DECAY_FACTOR * 0.55
        elif front_d < 1.2:
            sp_factor = self._DECAY_FACTOR
        else:
            sp_factor = 1.0

        if (right_corner or left_corner) and corner_allowed:
            sp_factor = min(sp_factor, 0.30)
        if raw_side_min < self._SIDE_SAFE_DIST:
            sp_factor = min(sp_factor, 0.45)

        # ========== 6. 三重保险实施 ==========
        vx = speed * direction * sp_factor
        vx = max(vx, 0.0)
        vx = min(vx, vx_max)

        w = clamp(w, -0.50, 0.50)
        w *= angular_scale

        if turn_locked:
            w = 0.0
            vx = min(vx, self._V_MAX * 0.5)

        cmd = Twist()
        cmd.linear.x = vx
        cmd.linear.y = ly
        cmd.angular.z = w
        return cmd

    def _reactive_cmd_reverse(
        self,
        speed: float,
        direction: float,
        corridor_bias: float,
        lateral_scale: float,
        angular_scale: float,
    ) -> Twist:
        """反向行进（返程）：使用简化居中跟随，不依赖左墙不对称判据。"""
        clearance = self._lidar.clearance()
        turn = self._lidar.preferred_turn()
        corridor_error = self._lidar.corridor_error() + corridor_bias

        cmd = Twist()
        cmd.linear.x = speed * direction
        cmd.linear.y = clamp(0.22 * corridor_error + 0.12 * turn, -0.18, 0.18) * lateral_scale
        cmd.angular.z = clamp(0.35 * turn, -0.45, 0.45) * angular_scale

        if clearance.has_data:
            if clearance.front < 0.55:
                cmd.linear.x = 0.04 * abs(direction)
                cmd.linear.y = clamp(cmd.linear.y + 0.18 * turn * lateral_scale, -0.22, 0.22)
            elif clearance.front < 0.9:
                cmd.linear.x *= 0.55

        if clearance.has_data and clearance.left > self._CORNER_DETECT_THRESH:
            cmd.linear.x *= 0.35
        return cmd

    def initial_direction_check(self, timeout: float = 8.0) -> bool:
        """利用墙/锥桶不对称判据判断正确前进方向。
        
        右侧=实心墙(高墙度), 左侧=离散锥桶(聚类点数多) → 右墙跟随方向正确。
        返回 True 表示方向正确，False 表示不确定或需要调整。
        """
        rospy.loginfo("=== 开局方向判断：利用墙/锥桶不对称性（右墙跟随） ===")
        deadline = time.time() + timeout
        rate = rospy.Rate(10)

        while not rospy.is_shutdown() and time.time() < deadline:
            c = self._lidar.clearance()
            if not c.has_data:
                rate.sleep()
                continue

            left_score = self._lidar.wall_score_left()
            right_score = self._lidar.wall_score_right()
            cone_dist, num_cones, _ = self._lidar.left_cone_info(
                cone_diameter=self._CONE_DIAMETER,
                cone_min_points=self._CONE_MIN_POINTS,
                sparsity_threshold=self._SPARSITY_THRESH,
            )

            right_is_wall = right_score > self._WALL_SCORE_THRESH
            left_has_cones = num_cones > 0

            rospy.loginfo(
                "方向判断: left_wall=%.2f right_wall=%.2f(%s) cones_left=%d(%.2fm)",
                left_score, right_score, "WALL" if right_is_wall else "open",
                num_cones, cone_dist,
            )

            if right_is_wall and left_has_cones:
                rospy.loginfo("方向正确：右侧为实心墙，左侧检测到锥桶")
                return True

            # 反向判据：左侧=墙 + 右侧空旷/无锥桶 → 方向相反
            left_is_wall = left_score > self._WALL_SCORE_THRESH
            if left_is_wall and right_score < 0.25:
                rospy.logwarn("方向疑似反转：左侧为实心墙，右侧空旷/锥桶")
                return False

            rate.sleep()

        rospy.logwarn("方向判断超时(%.1fs)，默认认为方向正确", timeout)
        return True

    def _detect_lit_button(self) -> Optional[str]:
        deadline = time.time() + 3.0
        while not rospy.is_shutdown() and time.time() < deadline:
            green = self._button_detector.is_lit("green")
            red = self._button_detector.is_lit("red")
            if green is True and red is not True:
                return "green"
            if red is True and green is not True:
                return "red"
            rospy.sleep(0.1)
        return None

    def _back_off(self, distance: float):
        cmd = Twist()
        cmd.linear.x = -0.10
        self._cmd_pub.publish(cmd)
        rospy.sleep(distance / 0.10)
        self.stop(0.3)
