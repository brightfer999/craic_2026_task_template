#!/usr/bin/env python3

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple

import rospy
from nav_msgs.msg import Odometry

from utils.math_utils import normalize_angle, quaternion_to_euler
from waypoints import APRILTAG_ANCHORS


@dataclass
class PoseEstimate:
    x: float
    y: float
    yaw: float
    stamp: rospy.Time
    valid: bool = True


@dataclass
class AnchorObservation:
    tag_id: int
    name: str
    stage_hint: str
    forward: float
    lateral: float
    distance: float
    stamp: rospy.Time


class Localizer:
    """Relative odometry tracker with optional AprilTag semantic anchors."""

    def __init__(self):
        self._lock = threading.Lock()
        self._latest_odom = None
        self._origin_odom = None
        self._stage_origin = None
        self._latest_anchor: Optional[AnchorObservation] = None
        self._odom_ready = False
        self._tag_sub = None
        self._odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        self._try_subscribe_apriltag()

    def _try_subscribe_apriltag(self):
        try:
            from apriltag_ros.msg import AprilTagDetectionArray

            self._tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self._tag_callback)
            rospy.loginfo("Localizer subscribed to /tag_detections")
        except Exception as exc:
            rospy.logwarn("AprilTag messages unavailable, using relative odom only: %s", exc)

    def init_localize(self, wait_timeout: float = 3.0) -> bool:
        """Wait briefly for odom; AprilTag is optional and never blocks startup."""
        deadline = rospy.Time.now() + rospy.Duration(wait_timeout)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < deadline:
            with self._lock:
                if self._latest_odom is not None:
                    self._origin_odom = self._extract_odom_pose_locked()
                    self._stage_origin = self._origin_odom
                    self._odom_ready = self._origin_odom is not None
                    break
            rate.sleep()

        if self._odom_ready:
            rospy.loginfo("Localizer initialized with relative odom; AprilTag anchor is optional")
            return True

        rospy.logwarn("Localizer did not receive /odom; patrol will still publish cautious commands")
        return False

    def get_pose(self) -> PoseEstimate:
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._origin_odom

        if current is None or origin is None:
            return PoseEstimate(0.0, 0.0, 0.0, rospy.Time.now(), False)

        dx = current[0] - origin[0]
        dy = current[1] - origin[1]
        yaw0 = origin[2]
        rel_x = math.cos(yaw0) * dx + math.sin(yaw0) * dy
        rel_y = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
        rel_yaw = normalize_angle(current[2] - yaw0)
        return PoseEstimate(rel_x, rel_y, rel_yaw, rospy.Time.now(), True)

    def reset_stage_progress(self):
        with self._lock:
            self._stage_origin = self._extract_odom_pose_locked()

    def distance_since_stage_reset(self) -> float:
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._stage_origin
        if current is None or origin is None:
            return 0.0
        return math.hypot(current[0] - origin[0], current[1] - origin[1])

    def yaw_since_stage_reset(self) -> float:
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._stage_origin
        if current is None or origin is None:
            return 0.0
        return normalize_angle(current[2] - origin[2])

    def latest_anchor(self, max_age: float = 2.0) -> Optional[AnchorObservation]:
        with self._lock:
            anchor = self._latest_anchor
        if anchor is None:
            return None
        if (rospy.Time.now() - anchor.stamp).to_sec() > max_age:
            return None
        return anchor

    def has_seen_tag(self, tag_ids: Tuple[int, ...], max_age: float = 3.0) -> bool:
        anchor = self.latest_anchor(max_age=max_age)
        return anchor is not None and anchor.tag_id in tag_ids

    @property
    def pose_is_valid(self) -> bool:
        return self.get_pose().valid

    @property
    def anchor_is_valid(self) -> bool:
        return self.latest_anchor() is not None

    def _odom_callback(self, msg):
        with self._lock:
            self._latest_odom = msg
            if self._origin_odom is None:
                self._origin_odom = self._extract_odom_pose_locked()
                self._stage_origin = self._origin_odom
                self._odom_ready = self._origin_odom is not None

    def _tag_callback(self, msg):
        best = None
        for detection in getattr(msg, "detections", []):
            tag_ids = getattr(detection, "id", [])
            if not tag_ids:
                continue
            tag_id = int(tag_ids[0])
            anchor = APRILTAG_ANCHORS.get(tag_id)
            if anchor is None:
                continue

            pose = detection.pose.pose.pose
            forward = float(pose.position.z)
            lateral = float(-pose.position.x)
            distance = math.hypot(forward, lateral)
            if distance > 3.0:
                continue

            obs = AnchorObservation(
                tag_id=tag_id,
                name=anchor.name,
                stage_hint=anchor.stage_hint,
                forward=forward,
                lateral=lateral,
                distance=distance,
                stamp=rospy.Time.now(),
            )
            if best is None or obs.distance < best.distance:
                best = obs

        if best is None:
            return

        with self._lock:
            self._latest_anchor = best
        rospy.loginfo_throttle(
            1.0,
            "AprilTag anchor: id=%d name=%s stage=%s dist=%.2f",
            best.tag_id,
            best.name,
            best.stage_hint,
            best.distance,
        )

    def _extract_odom_pose_locked(self) -> Optional[Tuple[float, float, float]]:
        if self._latest_odom is None:
            return None
        pose = self._latest_odom.pose.pose
        q = pose.orientation
        _roll, _pitch, yaw = quaternion_to_euler(q.x, q.y, q.z, q.w)
        return pose.position.x, pose.position.y, yaw
