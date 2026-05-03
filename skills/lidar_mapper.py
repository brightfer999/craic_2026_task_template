"""
激光雷达局部建图模块
提供机器人周围的实时局部地图和避障信息
"""

import math
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points


@dataclass
class Clearance:
    """机器人周围各方向的clearance信息"""
    front: float
    left_front: float
    right_front: float
    left: float
    right: float
    rear: float
    has_data: bool


class LidarMapper:
    """机器人中心的激光雷达局部建图器
    
    提供实时的局部地图信息，包括：
    - 各扇区clearance（距离最近障碍物的距离）
    - 走廊跟随误差
    - 推荐转向方向
    - 前方是否清晰
    """
    
    SECTORS = {
        "front": (-math.radians(18), math.radians(18)),
        "left_front": (math.radians(18), math.radians(68)),
        "right_front": (-math.radians(68), -math.radians(18)),
        "left": (math.radians(68), math.radians(120)),
        "right": (-math.radians(120), -math.radians(68)),
        "rear": (math.radians(150), math.pi),
    }
    
    def __init__(self, topic: str = "/lidar/points", max_range: float = 4.5):
        """初始化激光雷达建图器
        
        Args:
            topic: 激光雷达点云话题
            max_range: 最大检测距离（米）
        """
        self.max_range = max_range
        self._lock = threading.Lock()
        self._latest_cloud = None
        self._sub = rospy.Subscriber(topic, PointCloud2, self._cloud_callback)
        
    def _cloud_callback(self, msg):
        """点云回调函数"""
        with self._lock:
            self._latest_cloud = msg
            
    def get_points(self, max_points: int = 3000) -> List[Tuple[float, float, float]]:
        """获取过滤后的点云数据
        
        Args:
            max_points: 最大点数限制
            
        Returns:
            点云列表，每个点为(x, y, z)元组
        """
        with self._lock:
            cloud = self._latest_cloud
        if cloud is None:
            return []
            
        points = []
        for i, p in enumerate(read_points(cloud, field_names=("x", "y", "z"), skip_nans=True)):
            if i >= max_points:
                break
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            distance = math.hypot(x, y)
            # 过滤距离范围和高度范围
            if 0.12 <= distance <= self.max_range and -0.25 <= z <= 1.25:
                points.append((x, y, z))
        return points
        
    def clearance(self) -> Clearance:
        """计算机器人周围各扇区的clearance
        
        Returns:
            Clearance对象，包含各方向最近障碍物距离
        """
        distances: Dict[str, float] = {name: self.max_range for name in self.SECTORS}
        points = self.get_points()
        
        for x, y, _z in points:
            angle = math.atan2(y, x)
            distance = math.hypot(x, y)
            
            for name, (low, high) in self.SECTORS.items():
                if name == "rear":
                    in_sector = angle >= low or angle <= -low
                else:
                    in_sector = low <= angle <= high
                    
                if in_sector and distance < distances[name]:
                    distances[name] = distance
                    
        return Clearance(
            front=distances["front"],
            left_front=distances["left_front"],
            right_front=distances["right_front"],
            left=distances["left"],
            right=distances["right"],
            rear=distances["rear"],
            has_data=bool(points),
        )
        
    def front_clear(self, threshold: float = 0.75) -> bool:
        """检查前方是否清晰
        
        Args:
            threshold: 清晰阈值（米）
            
        Returns:
            前方距离大于阈值或无数据时返回True
        """
        c = self.clearance()
        return (not c.has_data) or c.front >= threshold
        
    def preferred_turn(self) -> float:
        """计算推荐的转向方向
        
        Returns:
            正值表示左转，负值表示右转，0表示直行
        """
        c = self.clearance()
        if not c.has_data:
            return 0.0
        if c.front > 1.1:
            return 0.0
            
        # 计算左右两侧的得分
        left_score = c.left_front + 0.5 * c.left
        right_score = c.right_front + 0.5 * c.right
        
        return 1.0 if left_score >= right_score else -1.0
        
    def corridor_error(self) -> float:
        """计算走廊跟随误差
        
        Returns:
            正值表示偏左，负值表示偏右，0表示居中
        """
        c = self.clearance()
        if not c.has_data:
            return 0.0
        if c.left >= self.max_range and c.right >= self.max_range:
            return 0.0
        if c.left >= self.max_range:
            return -0.25
        if c.right >= self.max_range:
            return 0.25
            
        # 计算左右偏差
        return max(-0.4, min(0.4, 0.5 * (c.right - c.left)))
        
    def obstacle_ahead(self, threshold: float = 0.55) -> bool:
        """检查前方是否有障碍物
        
        Args:
            threshold: 障碍物距离阈值（米）
            
        Returns:
            前方距离小于阈值时返回True
        """
        c = self.clearance()
        return c.has_data and c.front < threshold
        
    def get_obstacle_density(self, radius: float = 1.0) -> float:
        """计算指定半径内的障碍物密度
        
        Args:
            radius: 检测半径（米）
            
        Returns:
            障碍物点数密度（点数/平方米）
        """
        points = self.get_points()
        if not points:
            return 0.0
            
        # 统计半径内的点数
        count = 0
        for x, y, _ in points:
            if math.hypot(x, y) <= radius:
                count += 1
                
        # 计算密度
        area = math.pi * radius ** 2
        return count / area
        
    def find_clear_direction(self, num_sectors: int = 12, min_clearance: float = 1.0) -> Optional[float]:
        """找到最清晰的行驶方向
        
        Args:
            num_sectors: 扇区数量
            min_clearance: 最小clearance阈值（米）
            
        Returns:
            最清晰方向的角度（弧度），如果没有清晰方向返回None
        """
        points = self.get_points()
        if not points:
            return 0.0  # 无数据时默认前方
            
        # 将空间分成扇区
        sector_clearances = [0.0] * num_sectors
        sector_counts = [0] * num_sectors
        
        for x, y, _ in points:
            angle = math.atan2(y, x)
            distance = math.hypot(x, y)
            
            # 将角度映射到扇区索引
            sector_idx = int((angle + math.pi) / (2 * math.pi) * num_sectors) % num_sectors
            sector_clearances[sector_idx] += distance
            sector_counts[sector_idx] += 1
            
        # 计算每个扇区的平均距离
        avg_clearances = []
        for i in range(num_sectors):
            if sector_counts[i] > 0:
                avg_clearances.append(sector_clearances[i] / sector_counts[i])
            else:
                avg_clearances.append(self.max_range)  # 无数据时认为是清晰的
                
        # 找到最清晰的扇区
        best_sector = avg_clearances.index(max(avg_clearances))
        best_angle = (best_sector + 0.5) * (2 * math.pi) / num_sectors - math.pi
        
        # 检查是否满足最小clearance要求
        if avg_clearances[best_sector] >= min_clearance:
            return best_angle
        return None
        
    @property
    def has_data(self) -> bool:
        """检查是否有可用的点云数据"""
        with self._lock:
            return self._latest_cloud is not None