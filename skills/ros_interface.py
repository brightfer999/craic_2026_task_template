"""
ROS接口封装模块
提供统一的ROS接口访问方式
"""

import rospy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import JointState, PointCloud2, CompressedImage, CameraInfo
import numpy as np
from typing import Callable, Optional, Dict, Any

# 注意：以下导入需要ROS环境支持
# from kuavo_msgs.msg import sensorsData, switchController


class ROSInterface:
    """ROS接口封装类"""
    
    def __init__(self):
        """初始化ROS接口"""
        self.cmd_vel_pub = None
        self.arm_traj_pub = None
        self.gripper_pub = None
        self.switch_controller_client = None
        
        # 传感器数据存储
        self.sensor_data = {
            'odom': None,
            'lidar': None,
            'tag_detections': None,
            'cam_h_rgb': None,
            'cam_l_rgb': None,
            'cam_r_rgb': None,
            'cam_h_depth': None,
            'cam_l_depth': None,
            'cam_r_depth': None,
            'gripper_state': None,
            'sensors_data_raw': None
        }
        
        # 回调函数存储
        self.callbacks = {}
        
    def initialize_publishers(self):
        """初始化发布者"""
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.arm_traj_pub = rospy.Publisher('/kuavo_arm_traj', JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher('/gripper/command', JointState, queue_size=1)
        
    def initialize_subscribers(self):
        """初始化订阅者"""
        # 里程计
        rospy.Subscriber('/odom', ... , self._odom_callback)
        
        # 激光雷达
        rospy.Subscriber('/lidar/points', PointCloud2, self._lidar_callback)
        
        # AprilTag检测
        rospy.Subscriber('/tag_detections', ... , self._tag_callback)
        
        # 摄像头
        rospy.Subscriber('/cam_h/color/image_raw/compressed', CompressedImage, self._cam_h_rgb_callback)
        rospy.Subscriber('/cam_l/color/image_raw/compressed', CompressedImage, self._cam_l_rgb_callback)
        rospy.Subscriber('/cam_r/color/image_raw/compressed', CompressedImage, self._cam_r_rgb_callback)
        
        # 深度摄像头
        rospy.Subscriber('/cam_h/depth/image_raw/compressedDepth', CompressedImage, self._cam_h_depth_callback)
        rospy.Subscriber('/cam_l/depth/image_rect_raw/compressedDepth', CompressedImage, self._cam_l_depth_callback)
        rospy.Subscriber('/cam_r/depth/image_rect_raw/compressedDepth', CompressedImage, self._cam_r_depth_callback)
        
        # 夹爪状态
        rospy.Subscriber('/gripper/state', JointState, self._gripper_state_callback)
        
        # 传感器原始数据
        rospy.Subscriber('/sensors_data_raw', sensorsData, self._sensors_data_callback)
        
    def send_velocity_command(self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0):
        """发送速度命令"""
        twist = Twist()
        twist.linear.x = linear_x
        twist.linear.y = linear_y
        twist.angular.z = angular_z
        self.cmd_vel_pub.publish(twist)
        
    def send_arm_trajectory(self, joint_positions: list):
        """发送手臂轨迹"""
        msg = JointState()
        msg.position = joint_positions
        self.arm_traj_pub.publish(msg)
        
    def send_gripper_command(self, left_position: int, right_position: int):
        """发送夹爪命令"""
        msg = JointState()
        msg.position = [left_position, right_position]
        self.gripper_pub.publish(msg)
        
    def switch_controller(self, controller_name: str):
        """切换控制器"""
        # 实现控制器切换逻辑
        pass
        
    def get_sensor_data(self, sensor_name: str) -> Optional[Any]:
        """获取传感器数据"""
        return self.sensor_data.get(sensor_name)
        
    def register_callback(self, sensor_name: str, callback: Callable):
        """注册回调函数"""
        self.callbacks[sensor_name] = callback
        
    # 回调函数实现
    def _odom_callback(self, msg):
        self.sensor_data['odom'] = msg
        if 'odom' in self.callbacks:
            self.callbacks['odom'](msg)
            
    def _lidar_callback(self, msg):
        self.sensor_data['lidar'] = msg
        if 'lidar' in self.callbacks:
            self.callbacks['lidar'](msg)
            
    def _tag_callback(self, msg):
        self.sensor_data['tag_detections'] = msg
        if 'tag_detections' in self.callbacks:
            self.callbacks['tag_detections'](msg)
            
    def _cam_h_rgb_callback(self, msg):
        self.sensor_data['cam_h_rgb'] = msg
        if 'cam_h_rgb' in self.callbacks:
            self.callbacks['cam_h_rgb'](msg)
            
    def _cam_l_rgb_callback(self, msg):
        self.sensor_data['cam_l_rgb'] = msg
        if 'cam_l_rgb' in self.callbacks:
            self.callbacks['cam_l_rgb'](msg)
            
    def _cam_r_rgb_callback(self, msg):
        self.sensor_data['cam_r_rgb'] = msg
        if 'cam_r_rgb' in self.callbacks:
            self.callbacks['cam_r_rgb'](msg)
            
    def _cam_h_depth_callback(self, msg):
        self.sensor_data['cam_h_depth'] = msg
        if 'cam_h_depth' in self.callbacks:
            self.callbacks['cam_h_depth'](msg)
            
    def _cam_l_depth_callback(self, msg):
        self.sensor_data['cam_l_depth'] = msg
        if 'cam_l_depth' in self.callbacks:
            self.callbacks['cam_l_depth'](msg)
            
    def _cam_r_depth_callback(self, msg):
        self.sensor_data['cam_r_depth'] = msg
        if 'cam_r_depth' in self.callbacks:
            self.callbacks['cam_r_depth'](msg)
            
    def _gripper_state_callback(self, msg):
        self.sensor_data['gripper_state'] = msg
        if 'gripper_state' in self.callbacks:
            self.callbacks['gripper_state'](msg)
            
    def _sensors_data_callback(self, msg):
        self.sensor_data['sensors_data_raw'] = msg
        if 'sensors_data_raw' in self.callbacks:
            self.callbacks['sensors_data_raw'](msg)