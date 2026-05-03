"""
按钮检测模块
提供基于视觉的按钮颜色检测功能
"""

import rospy
import cv2
import numpy as np
from typing import Optional, Tuple, List
from sensor_msgs.msg import CompressedImage
from cv_bridge import CvBridge


class ButtonDetector:
    """按钮颜色检测器
    
    使用HSV颜色空间检测按钮颜色，支持红色、绿色、蓝色、黄色按钮。
    可用于检测操作面板上的按钮状态。
    """
    
    # HSV颜色范围定义
    COLOR_RANGES = {
        "red": (np.array([0, 120, 70]), np.array([10, 255, 255])),
        "green": (np.array([36, 50, 70]), np.array([86, 255, 255])),
        "blue": (np.array([94, 80, 2]), np.array([126, 255, 255])),
        "yellow": (np.array([20, 100, 100]), np.array([30, 255, 255])),
    }
    
    def __init__(self, camera_topic: str = "/cam_h/color/image_raw/compressed"):
        """初始化按钮检测器
        
        Args:
            camera_topic: 摄像头话题名称
        """
        self._bridge = CvBridge()
        self._latest_frame = None
        self._camera_topic = camera_topic
        self._sub = None
        
        # 订阅摄像头话题
        try:
            self._sub = rospy.Subscriber(camera_topic, CompressedImage, self._image_callback)
            rospy.loginfo(f"ButtonDetector subscribed to {camera_topic}")
        except Exception as e:
            rospy.logwarn(f"Failed to subscribe to {camera_topic}: {e}")
            
    def _image_callback(self, msg):
        """图像回调函数"""
        try:
            self._latest_frame = self._bridge.compressed_imgmsg_to_cv2(msg, "bgr8")
        except Exception:
            try:
                self._latest_frame = self._bridge.imgmsg_to_cv2(msg, "bgr8")
            except Exception as e:
                rospy.logwarn(f"Image conversion failed: {e}")
                
    def detect_button(self, color_name: str = "red") -> Optional[Tuple[int, int, float]]:
        """检测指定颜色的按钮
        
        Args:
            color_name: 颜色名称（red, green, blue, yellow）
            
        Returns:
            检测到按钮时返回(cx, cy, radius)，否则返回None
        """
        if self._latest_frame is None or color_name not in self.COLOR_RANGES:
            return None
            
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.COLOR_RANGES[color_name]
        
        # 创建颜色掩膜
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=4)
        
        # 查找轮廓
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None
            
        # 找到最大的轮廓
        largest = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest) < 500:  # 最小面积阈值
            return None
            
        # 计算中心点和半径
        moments = cv2.moments(largest)
        if moments["m00"] == 0:
            return None
            
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
        radius = np.sqrt(cv2.contourArea(largest) / np.pi)
        
        return cx, cy, radius
        
    def is_lit(self, color_name: str = "red", threshold: float = 0.15) -> Optional[bool]:
        """检查指定颜色的按钮是否亮起
        
        Args:
            color_name: 颜色名称
            threshold: 亮起阈值（0-1）
            
        Returns:
            检测成功时返回True/False，检测失败返回None
        """
        if self._latest_frame is None or color_name not in self.COLOR_RANGES:
            return None
            
        # 转换为HSV颜色空间
        hsv = cv2.cvtColor(self._latest_frame, cv2.COLOR_BGR2HSV)
        lower, upper = self.COLOR_RANGES[color_name]
        
        # 创建颜色掩膜
        mask = cv2.inRange(hsv, lower, upper)
        
        # 计算亮起比例
        lit_ratio = np.count_nonzero(mask) / mask.size
        return lit_ratio > threshold
        
    def detect_all_buttons(self) -> List[dict]:
        """检测所有支持颜色的按钮
        
        Returns:
            检测到的按钮列表，每个按钮包含颜色、位置和半径
        """
        buttons = []
        for color_name in self.COLOR_RANGES.keys():
            result = self.detect_button(color_name)
            if result:
                cx, cy, radius = result
                buttons.append({
                    "color": color_name,
                    "position": (cx, cy),
                    "radius": radius,
                    "is_lit": self.is_lit(color_name)
                })
        return buttons
        
    def get_button_color_at(self, x: int, y: int, radius: int = 20) -> Optional[str]:
        """获取指定位置的按钮颜色
        
        Args:
            x: 图像x坐标
            y: 图像y坐标
            radius: 检测半径（像素）
            
        Returns:
            颜色名称，如果无法识别返回None
        """
        if self._latest_frame is None:
            return None
            
        # 提取感兴趣区域
        h, w = self._latest_frame.shape[:2]
        x1 = max(0, x - radius)
        y1 = max(0, y - radius)
        x2 = min(w, x + radius)
        y2 = min(h, y + radius)
        
        roi = self._latest_frame[y1:y2, x1:x2]
        if roi.size == 0:
            return None
            
        # 转换为HSV并计算颜色直方图
        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        
        # 检查每种颜色的匹配程度
        best_color = None
        best_score = 0
        
        for color_name, (lower, upper) in self.COLOR_RANGES.items():
            mask = cv2.inRange(hsv_roi, lower, upper)
            score = np.count_nonzero(mask) / mask.size
            
            if score > best_score and score > 0.1:  # 最小匹配阈值
                best_score = score
                best_color = color_name
                
        return best_color
        
    def save_debug_image(self, filename: str = "debug_button.jpg"):
        """保存调试图像
        
        Args:
            filename: 保存文件名
        """
        if self._latest_frame is not None:
            cv2.imwrite(filename, self._latest_frame)
            rospy.loginfo(f"Debug image saved to {filename}")
            
    @property
    def has_frame(self) -> bool:
        """检查是否有可用的图像帧"""
        return self._latest_frame is not None
        
    @property
    def frame_shape(self) -> Optional[Tuple[int, int, int]]:
        """获取图像帧的形状"""
        if self._latest_frame is not None:
            return self._latest_frame.shape
        return None