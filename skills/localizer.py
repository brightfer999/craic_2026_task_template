"""
定位模块
提供相对里程计和AprilTag语义锚点定位功能
"""

import math
import threading
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import rospy
from nav_msgs.msg import Odometry


@dataclass
class PoseEstimate:
    """位姿估计"""
    x: float
    y: float
    yaw: float
    stamp: rospy.Time
    valid: bool = True


@dataclass
class AnchorObservation:
    """AprilTag锚点观测"""
    tag_id: int
    name: str
    stage_hint: str
    forward: float
    lateral: float
    distance: float
    stamp: rospy.Time


class Localizer:
    """相对里程计跟踪器，支持可选的AprilTag语义锚点
    
    提供：
    - 相对里程计跟踪（基于odom）
    - AprilTag语义锚点检测
    - 阶段内行程跟踪
    """
    
    def __init__(self, apriltag_anchors: Optional[Dict[int, dict]] = None):
        """初始化定位器
        
        Args:
            apriltag_anchors: AprilTag锚点配置字典，格式为{tag_id: {name, stage_hint, ...}}
        """
        self._lock = threading.Lock()
        self._latest_odom = None
        self._origin_odom = None
        self._stage_origin = None
        self._latest_anchor: Optional[AnchorObservation] = None
        self._odom_ready = False
        self._tag_sub = None
        
        # AprilTag锚点配置
        self._apriltag_anchors = apriltag_anchors or {}
        
        # 订阅里程计
        self._odom_sub = rospy.Subscriber("/odom", Odometry, self._odom_callback)
        
        # 尝试订阅AprilTag
        self._try_subscribe_apriltag()
        
    def _try_subscribe_apriltag(self):
        """尝试订阅AprilTag话题"""
        try:
            from apriltag_ros.msg import AprilTagDetectionArray
            self._tag_sub = rospy.Subscriber("/tag_detections", AprilTagDetectionArray, self._tag_callback)
            rospy.loginfo("Localizer subscribed to /tag_detections")
        except Exception as exc:
            rospy.logwarn("AprilTag messages unavailable, using relative odom only: %s", exc)
            
    def init_localize(self, wait_timeout: float = 3.0) -> bool:
        """初始化定位，等待odom数据
        
        Args:
            wait_timeout: 等待超时时间（秒）
            
        Returns:
            是否成功初始化
        """
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
        """获取当前相对位姿
        
        Returns:
            相对于原点的位姿估计
        """
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._origin_odom
            
        if current is None or origin is None:
            return PoseEstimate(0.0, 0.0, 0.0, rospy.Time.now(), False)
            
        # 计算相对位姿
        dx = current[0] - origin[0]
        dy = current[1] - origin[1]
        yaw0 = origin[2]
        
        # 转换到相对坐标系
        rel_x = math.cos(yaw0) * dx + math.sin(yaw0) * dy
        rel_y = -math.sin(yaw0) * dx + math.cos(yaw0) * dy
        rel_yaw = self._normalize_angle(current[2] - yaw0)
        
        return PoseEstimate(rel_x, rel_y, rel_yaw, rospy.Time.now(), True)
        
    def reset_stage_progress(self):
        """重置阶段进度，将当前位置设为阶段起点"""
        with self._lock:
            self._stage_origin = self._extract_odom_pose_locked()
            
    def distance_since_stage_reset(self) -> float:
        """获取自阶段重置以来的行驶距离
        
        Returns:
            行驶距离（米）
        """
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._stage_origin
            
        if current is None or origin is None:
            return 0.0
            
        return math.hypot(current[0] - origin[0], current[1] - origin[1])
        
    def yaw_since_stage_reset(self) -> float:
        """获取自阶段重置以来的偏航角变化
        
        Returns:
            偏航角变化（弧度）
        """
        with self._lock:
            current = self._extract_odom_pose_locked()
            origin = self._stage_origin
            
        if current is None or origin is None:
            return 0.0
            
        return self._normalize_angle(current[2] - origin[2])
        
    def latest_anchor(self, max_age: float = 2.0) -> Optional[AnchorObservation]:
        """获取最新的AprilTag锚点观测
        
        Args:
            max_age: 最大有效时间（秒）
            
        Returns:
            锚点观测，如果过期或不存在返回None
        """
        with self._lock:
            anchor = self._latest_anchor
            
        if anchor is None:
            return None
            
        if (rospy.Time.now() - anchor.stamp).to_sec() > max_age:
            return None
            
        return anchor
        
    def has_seen_tag(self, tag_ids: Tuple[int, ...], max_age: float = 3.0) -> bool:
        """检查是否看到指定的AprilTag
        
        Args:
            tag_ids: 标签ID元组
            max_age: 最大有效时间（秒）
            
        Returns:
            是否看到指定标签
        """
        anchor = self.latest_anchor(max_age=max_age)
        return anchor is not None and anchor.tag_id in tag_ids
        
    def get_distance_to_anchor(self, tag_id: int) -> Optional[float]:
        """获取到指定AprilTag锚点的距离
        
        Args:
            tag_id: 标签ID
            
        Returns:
            距离（米），如果未检测到返回None
        """
        anchor = self.latest_anchor()
        if anchor and anchor.tag_id == tag_id:
            return anchor.distance
        return None
        
    def get_direction_to_anchor(self, tag_id: int) -> Optional[Tuple[float, float]]:
        """获取到指定AprilTag锚点的方向
        
        Args:
            tag_id: 标签ID
            
        Returns:
            (forward, lateral)方向，如果未检测到返回None
        """
        anchor = self.latest_anchor()
        if anchor and anchor.tag_id == tag_id:
            return (anchor.forward, anchor.lateral)
        return None
        
    @property
    def pose_is_valid(self) -> bool:
        """检查位姿是否有效"""
        return self.get_pose().valid
        
    @property
    def anchor_is_valid(self) -> bool:
        """检查锚点是否有效"""
        return self.latest_anchor() is not None
        
    @property
    def is_ready(self) -> bool:
        """检查定位器是否就绪"""
        return self._odom_ready
        
    def _odom_callback(self, msg):
        """里程计回调函数"""
        with self._lock:
            self._latest_odom = msg
            if self._origin_odom is None:
                self._origin_odom = self._extract_odom_pose_locked()
                self._stage_origin = self._origin_odom
                self._odom_ready = self._origin_odom is not None
                
    def _tag_callback(self, msg):
        """AprilTag回调函数"""
        best = None
        for detection in getattr(msg, "detections", []):
            tag_ids = getattr(detection, "id", [])
            if not tag_ids:
                continue
                
            tag_id = int(tag_ids[0])
            anchor_config = self._apriltag_anchors.get(tag_id)
            if anchor_config is None:
                continue
                
            # 提取位姿信息
            pose = detection.pose.pose.pose
            forward = float(pose.position.z)
            lateral = float(-pose.position.x)
            distance = math.hypot(forward, lateral)
            
            # 距离过滤
            if distance > 3.0:
                continue
                
            # 创建观测
            obs = AnchorObservation(
                tag_id=tag_id,
                name=anchor_config.get("name", f"tag_{tag_id}"),
                stage_hint=anchor_config.get("stage_hint", ""),
                forward=forward,
                lateral=lateral,
                distance=distance,
                stamp=rospy.Time.now(),
            )
            
            # 选择最近的标签
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
        """提取里程计位姿（需要持有锁）"""
        if self._latest_odom is None:
            return None
            
        pose = self._latest_odom.pose.pose
        q = pose.orientation
        _roll, _pitch, yaw = self._quaternion_to_euler(q.x, q.y, q.z, q.w)
        
        return pose.position.x, pose.position.y, yaw
        
    @staticmethod
    def _normalize_angle(angle: float) -> float:
        """将角度标准化到[-pi, pi]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
        
    @staticmethod
    def _quaternion_to_euler(qx: float, qy: float, qz: float, qw: float) -> Tuple[float, float, float]:
        """四元数转欧拉角"""
        # Roll (x轴旋转)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2.0 * (qw * qy - qz * qx)
        pitch = math.asin(max(-1.0, min(1.0, sinp)))
        
        # Yaw (z轴旋转)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw