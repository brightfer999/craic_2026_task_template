#!/usr/bin/env python3

import time
from typing import Optional

import rospy
from geometry_msgs.msg import Twist

from button_presser import ButtonDetector, ButtonPresser
from controller_manager import ControllerManager
from local_lidar_map import LocalLidarMapper
from localization import Localizer
from utils.math_utils import clamp
from waypoints import BUTTON_PANELS, StageDirective


class TerrainTraverser:
    """Perception-driven movement primitives for scene 1."""

    def __init__(self, localizer: Localizer, controller: ControllerManager):
        self._localizer = localizer
        self._controller = controller
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._lidar = LocalLidarMapper()
        self._button_detector = ButtonDetector()
        self._button_presser = ButtonPresser()
        self.obstacle_hit_count = 0

    def run_stage(self, stage: StageDirective, reverse: bool = False) -> bool:
        if stage.controller == "amp":
            self._controller.use_amp()
        else:
            self._controller.use_mpc()

        if stage.name in ("operator_1", "operator_2"):
            return self.press_panel(stage.name)
        if stage.name == "goal":
            ok = self.advance_until_tag_or_distance(stage, reverse=reverse, stop_on_tag=True)
            self.stop(3.0)
            return ok
        if "obstacle" in stage.name:
            return self.cross_obstacle_zone(stage, reverse=reverse)
        return self.follow_corridor(stage, reverse=reverse)

    def advance_until_tag_or_distance(
        self,
        stage: StageDirective,
        reverse: bool = False,
        stop_on_tag: bool = False,
    ) -> bool:
        self._localizer.reset_stage_progress()
        start = time.time()
        rate = rospy.Rate(20)
        direction = -1.0 if reverse else 1.0
        saw_tag = False

        while not rospy.is_shutdown() and time.time() - start < stage.timeout:
            travelled = self._localizer.distance_since_stage_reset()
            if stage.target_tags and self._localizer.has_seen_tag(stage.target_tags):
                saw_tag = True
            if stage.completion_tags and self._localizer.has_seen_tag(stage.completion_tags):
                saw_tag = True
                if stop_on_tag:
                    self.stop(0.4)
                    return True
            if travelled >= stage.nominal_distance:
                self.stop(0.4)
                return True

            cmd = self._reactive_cmd(stage.speed, direction, stage.corridor_bias)
            self._cmd_pub.publish(cmd)
            rate.sleep()

        self.stop()
        rospy.logwarn("%s timed out after %.1fm, saw_tag=%s", stage.name, self._localizer.distance_since_stage_reset(), saw_tag)
        return saw_tag or self._localizer.distance_since_stage_reset() >= 0.7 * stage.nominal_distance

    def cross_obstacle_zone(self, stage: StageDirective, reverse: bool = False) -> bool:
        self._localizer.reset_stage_progress()
        start = time.time()
        rate = rospy.Rate(20)
        direction = -1.0 if reverse else 1.0

        while not rospy.is_shutdown() and time.time() - start < stage.timeout:
            travelled = self._localizer.distance_since_stage_reset()
            if travelled >= stage.nominal_distance:
                self.stop(0.5)
                return True

            clearance = self._lidar.clearance()
            if clearance.has_data and clearance.front < 0.42:
                self.obstacle_hit_count += 1

            cmd = self._reactive_cmd(stage.speed, direction, stage.corridor_bias)
            if clearance.has_data and clearance.front < 0.75:
                cmd.linear.x = clamp(cmd.linear.x, -0.08, 0.08)
            self._cmd_pub.publish(cmd)
            rate.sleep()

        self.stop()
        rospy.logwarn("Obstacle stage timed out after %.1fm", self._localizer.distance_since_stage_reset())
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

    def _reactive_cmd(self, speed: float, direction: float, corridor_bias: float = 0.0) -> Twist:
        clearance = self._lidar.clearance()
        turn = self._lidar.preferred_turn()
        corridor_error = self._lidar.corridor_error() + corridor_bias

        cmd = Twist()
        cmd.linear.x = speed * direction
        cmd.linear.y = clamp(0.22 * corridor_error + 0.12 * turn, -0.18, 0.18)
        cmd.angular.z = clamp(0.35 * turn, -0.45, 0.45)

        if clearance.has_data:
            if clearance.front < 0.55:
                cmd.linear.x = 0.04 * direction
                cmd.linear.y = clamp(cmd.linear.y + 0.18 * turn, -0.22, 0.22)
            elif clearance.front < 0.9:
                cmd.linear.x *= 0.55
        return cmd

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
