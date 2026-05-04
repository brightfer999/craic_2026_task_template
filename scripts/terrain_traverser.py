#!/usr/bin/env python3

import math
import time
from typing import Optional

import rospy
from geometry_msgs.msg import Twist

from button_presser import ButtonDetector, ButtonPresser
from controller_manager import ControllerManager
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
        self._CORNER_DETECT_THRESH = float(rospy.get_param("~corner_detect_threshold", 2.5))
        self._CORNER_TURN_W = float(rospy.get_param("~corner_turn_w", 0.35))
        self._LEFT_CORNER_TURN_W = float(rospy.get_param("~left_corner_turn_w", 0.35))
        self._DECAY_FACTOR = float(rospy.get_param("~decay_factor", 0.50))
        self._V_MAX = float(rospy.get_param("~v_max", 0.25))
        self._WALL_SCORE_THRESH = float(rospy.get_param("~wall_score_threshold", 0.50))
        self._SPARSITY_THRESH = float(rospy.get_param("~sparsity_threshold", 0.35))
        self._MAX_TURN_ANGLE = math.radians(float(rospy.get_param("~max_turn_angle_deg", 126.0)))
        self._initial_yaw = None

        # ====== 锥桶看门狗：巡逻中未发现锥桶则掉头 ======
        self._cones_confirmed = False
        self.no_cone_abort = False
        self._cone_abort_distance = float(rospy.get_param("~no_cone_reversal_distance", 3.0))

    def perform_startup_lidar_rotation(
        self, duration: Optional[float] = None, angular_z: Optional[float] = None
    ) -> None:
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

        while not rospy.is_shutdown() and time.time() < deadline:
            self._cmd_pub.publish(cmd)
            rate.sleep()
            if time.time() - _last_watchdog >= 0.8:
                if self._check_cones_watchdog():
                    self.stop(0.3)
                    return False
                _last_watchdog = time.time()

        self.stop()
        rospy.logwarn(
            "%s timed out after %.1fm, saw_tag=%s",
            stage.name,
            self._localizer.distance_since_stage_reset(),
            saw_tag,
        )
        return saw_tag or self._localizer.distance_since_stage_reset() >= 0.7 * stage.nominal_distance

    def cross_obstacle_zone(self, stage: StageDirective, reverse: bool = False) -> bool:
        self._localizer.reset_stage_progress()
        self._initial_yaw = None  # 重置累计转弯角度跟踪
        start = time.time()
        direction = -1.0 if reverse else 1.0
        self._run_lidar_warmup_gate(direction)
        motion_t0 = time.time()
        rate = rospy.Rate(20)
        blend_motion = True
        _last_watchdog = time.time()

        while not rospy.is_shutdown() and time.time() - start < stage.timeout:
            travelled = self._localizer.distance_since_stage_reset()
            if travelled >= stage.nominal_distance:
                self.stop(0.5)
                return True

            clearance = self._lidar.clearance()
            if clearance.has_data and clearance.front < 0.42:
                self.obstacle_hit_count += 1

            la_scale, yz_scale = self._lateral_angular_blend(time.time() - motion_t0, blend_motion)
            cmd = self._reactive_cmd(
                stage.speed,
                direction,
                stage.corridor_bias,
                lateral_scale=la_scale,
                angular_scale=yz_scale,
                reverse=reverse,
            )
            if clearance.has_data and clearance.front < 0.75:
                cmd.linear.x = clamp(cmd.linear.x, -0.08, 0.08)
            self._cmd_pub.publish(cmd)
            rate.sleep()
            if time.time() - _last_watchdog >= 0.8:
                if self._check_cones_watchdog():
                    self.stop(0.3)
                    return False
                _last_watchdog = time.time()

        self.stop()
        rospy.logwarn(
            "Obstacle stage timed out after %.1fm",
            self._localizer.distance_since_stage_reset(),
        )
        return self._localizer.distance_since_stage_reset() >= 0.65 * stage.nominal_distance

    def follow_corridor(self, stage: StageDirective, reverse: bool = False) -> bool:
        return self.advance_until_tag_or_distance(stage, reverse=reverse, stop_on_tag=False)

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
        left_obs = clearance.left if has else self._lidar.max_range
        front_d = clearance.front if has else self._lidar.max_range

        # ========== 1. 右墙跟随 ==========
        wall_err = right_wall - self._WALL_FOLLOW_DISTANCE
        ly = -clamp(self._WALL_FOLLOW_KP_LAT * wall_err, -0.18, 0.18) * lateral_scale
        w_wall = clamp(-self._WALL_FOLLOW_KP_ANG * wall_err, 0.0, 0.35)

        # ========== 2. 左锥桶/障碍避让 ==========
        if left_obs < self._CONE_SAFE_DIST:
            factor = clamp(left_obs / self._CONE_SAFE_DIST, 0.0, 1.0)
            if w_wall > 0:
                w_wall *= factor

        # ========== 3. 双拐角检测 ==========
        right_corner = self._check_corner(clearance)
        left_corner = self._check_left_corner(clearance)

        if right_corner:
            w = self._CORNER_TURN_W
            front_d = min(front_d, 0.45)
            rospy.loginfo_throttle(1.0, "检测到右拐角，执行左转弯(避开场景二) w=%.2f", w)
        elif left_corner:
            w = self._LEFT_CORNER_TURN_W
            front_d = min(front_d, 0.45)
            rospy.loginfo_throttle(1.0, "检测到左拐角，执行左转弯 w=%.2f", w)
        else:
            w = w_wall

        # ========== 3.5. 双级避障（raw clearance 无 EMA 延迟）==========
        raw_c = self._lidar.raw_clearance()
        if raw_c.has_data:
            raw_min = min(raw_c.front, raw_c.right_front, raw_c.left_front,
                          raw_c.right * 0.6)
        else:
            raw_min = 99.0

        if raw_min < 0.20:
            # 极限：后退 + 猛左转
            cmd = Twist()
            cmd.linear.x = -0.06
            cmd.linear.y = 0.0
            cmd.angular.z = 0.50
            rospy.loginfo_throttle(1.0, "极限避障(%.2fm)，后退+左转", raw_min)
            return cmd

        if raw_min < 0.40:
            # 危险：硬停车 + 猛左转
            cmd = Twist()
            cmd.linear.x = 0.0
            cmd.linear.y = 0.0
            cmd.angular.z = 0.50
            rospy.loginfo_throttle(1.0, "避障刹车(%.2fm)，停车+左转", raw_min)
            return cmd

        # 用 raw 值加速速度衰减（EMA 有延迟）
        front_d = min(front_d, raw_min)

        # ========== 4. 速度衰减（前方越近越慢）==========
        if front_d < 0.45:
            sp_factor = 0.12
        elif front_d < 0.75:
            sp_factor = self._DECAY_FACTOR * 0.55
        elif front_d < 1.2:
            sp_factor = self._DECAY_FACTOR
        else:
            sp_factor = 1.0

        if right_corner or left_corner:
            sp_factor = min(sp_factor, 0.30)

        # ========== 5. 三重保险实施 ==========
        vx = speed * direction * sp_factor
        vx = max(vx, 0.0)
        vx = min(vx, self._V_MAX)

        w = clamp(w, -0.50, 0.50)
        w *= angular_scale

        if self._initial_yaw is not None:
            current_yaw = self._localizer.yaw_since_stage_reset()
            if abs(current_yaw) > self._MAX_TURN_ANGLE:
                w = 0.0
                vx = min(vx, self._V_MAX * 0.5)
                rospy.logwarn_throttle(0.5, "累计转弯超过%.1f°，锁定方向",
                                       math.degrees(self._MAX_TURN_ANGLE))
        else:
            self._initial_yaw = self._localizer.yaw_since_stage_reset()

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
