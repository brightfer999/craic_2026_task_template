"""
感知工具模块
提供感知相关的工具函数
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
import cv2
from sensor_msgs.msg import PointCloud2, CompressedImage
import sensor_msgs.point_cloud2 as pc2
from cv_bridge import CvBridge


class PerceptionUtils:
    """感知工具类"""
    
    def __init__(self):
        """初始化感知工具"""
        self.bridge = CvBridge()
        
    def point_cloud_to_array(self, cloud_msg: PointCloud2) -> np.ndarray:
        """将点云消息转换为numpy数组"""
        points = []
        for p in pc2.read_points(cloud_msg, skip_nans=True):
            points.append([p[0], p[1], p[2]])
        return np.array(points)
    
    def filter_point_cloud(self, points: np.ndarray, 
                          x_range: Tuple[float, float] = (-10, 10),
                          y_range: Tuple[float, float] = (-10, 10),
                          z_range: Tuple[float, float] = (-1, 2)) -> np.ndarray:
        """过滤点云数据"""
        mask = (
            (points[:, 0] >= x_range[0]) & (points[:, 0] <= x_range[1]) &
            (points[:, 1] >= y_range[0]) & (points[:, 1] <= y_range[1]) &
            (points[:, 2] >= z_range[0]) & (points[:, 2] <= z_range[1])
        )
        return points[mask]
    
    def downsample_point_cloud(self, points: np.ndarray, voxel_size: float = 0.1) -> np.ndarray:
        """下采样点云"""
        if len(points) == 0:
            return points
            
        # 计算体素网格
        min_coords = np.min(points, axis=0)
        max_coords = np.max(points, axis=0)
        
        # 计算每个维度的体素数量
        voxel_counts = np.ceil((max_coords - min_coords) / voxel_size).astype(int)
        
        # 为每个点分配体素索引
        voxel_indices = np.floor((points - min_coords) / voxel_size).astype(int)
        
        # 使用字典存储每个体素的点
        voxel_dict = {}
        for i, idx in enumerate(voxel_indices):
            key = tuple(idx)
            if key not in voxel_dict:
                voxel_dict[key] = []
            voxel_dict[key].append(points[i])
        
        # 对每个体素内的点取平均
        downsampled = []
        for voxel_points in voxel_dict.values():
            downsampled.append(np.mean(voxel_points, axis=0))
        
        return np.array(downsampled)
    
    def extract_obstacles(self, points: np.ndarray, 
                         height_threshold: float = 0.1,
                         min_points: int = 10) -> List[Dict[str, Any]]:
        """从点云中提取障碍物"""
        if len(points) == 0:
            return []
        
        # 简单的障碍物提取：基于高度阈值
        ground_mask = points[:, 2] < height_threshold
        obstacle_points = points[~ground_mask]
        
        if len(obstacle_points) == 0:
            return []
        
        # 使用简单的聚类方法
        obstacles = []
        visited = set()
        
        for i in range(len(obstacle_points)):
            if i in visited:
                continue
                
            # BFS聚类
            cluster = []
            queue = [i]
            visited.add(i)
            
            while queue:
                current = queue.pop(0)
                cluster.append(obstacle_points[current])
                
                # 查找邻近点
                for j in range(len(obstacle_points)):
                    if j in visited:
                        continue
                        
                    distance = np.linalg.norm(obstacle_points[current] - obstacle_points[j])
                    if distance < 0.5:  # 邻域半径
                        queue.append(j)
                        visited.add(j)
            
            if len(cluster) >= min_points:
                cluster_array = np.array(cluster)
                obstacle = {
                    'center': np.mean(cluster_array, axis=0),
                    'size': np.max(cluster_array, axis=0) - np.min(cluster_array, axis=0),
                    'points': cluster_array,
                    'num_points': len(cluster)
                }
                obstacles.append(obstacle)
        
        return obstacles
    
    def compressed_image_to_cv2(self, compressed_msg: CompressedImage) -> np.ndarray:
        """将压缩图像消息转换为OpenCV格式"""
        try:
            # 尝试直接解码
            np_arr = np.frombuffer(compressed_msg.data, np.uint8)
            image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
            return image
        except Exception as e:
            rospy.logerr(f"图像解码失败: {e}")
            return None
    
    def detect_apriltag(self, image: np.ndarray, 
                       tag_family: str = "tag36h11") -> List[Dict[str, Any]]:
        """检测图像中的AprilTag"""
        # 这里需要集成AprilTag检测库
        # 简化实现，返回空列表
        return []
    
    def calculate_obstacle_density(self, points: np.ndarray, 
                                  center: Tuple[float, float],
                                  radius: float = 1.0) -> float:
        """计算指定区域内的障碍物密度"""
        if len(points) == 0:
            return 0.0
            
        # 计算点到中心的距离
        distances = np.sqrt((points[:, 0] - center[0])**2 + (points[:, 1] - center[1])**2)
        
        # 统计半径内的点数
        points_in_radius = np.sum(distances <= radius)
        
        # 计算密度（点数/面积）
        area = np.pi * radius**2
        density = points_in_radius / area
        
        return density
    
    def find_clear_path(self, points: np.ndarray, 
                       start: Tuple[float, float],
                       end: Tuple[float, float],
                       width: float = 0.5) -> bool:
        """检查两点之间是否有清晰路径"""
        if len(points) == 0:
            return True
            
        # 计算路径方向
        direction = np.array([end[0] - start[0], end[1] - start[1]])
        length = np.linalg.norm(direction)
        
        if length < 1e-6:
            return True
            
        direction = direction / length
        
        # 计算垂直方向
        perpendicular = np.array([-direction[1], direction[0]])
        
        # 检查路径上的点
        for point in points:
            # 计算点到路径的距离
            point_vector = np.array([point[0] - start[0], point[1] - start[1]])
            
            # 投影到路径方向
            projection = np.dot(point_vector, direction)
            
            # 检查是否在路径范围内
            if 0 <= projection <= length:
                # 计算横向距离
                lateral_distance = abs(np.dot(point_vector, perpendicular))
                
                if lateral_distance < width / 2:
                    return False
        
        return True
    
    def calculate_traversability(self, points: np.ndarray, 
                                position: Tuple[float, float],
                                radius: float = 1.0) -> float:
        """计算指定位置的可通行性"""
        if len(points) == 0:
            return 1.0
            
        # 过滤区域内的点
        distances = np.sqrt((points[:, 0] - position[0])**2 + (points[:, 1] - position[1])**2)
        local_points = points[distances <= radius]
        
        if len(local_points) == 0:
            return 1.0
        
        # 计算高度标准差
        height_std = np.std(local_points[:, 2])
        
        # 计算坡度
        if len(local_points) >= 3:
            # 使用最小二乘法拟合平面
            A = np.column_stack([local_points[:, 0], local_points[:, 1], np.ones(len(local_points))])
            b = local_points[:, 2]
            try:
                plane_params = np.linalg.lstsq(A, b, rcond=None)[0]
                normal = np.array([plane_params[0], plane_params[1], -1])
                normal = normal / np.linalg.norm(normal)
                slope = np.arccos(abs(normal[2]))
            except:
                slope = 0.0
        else:
            slope = 0.0
        
        # 综合评估可通行性
        # 高度标准差越大，可通行性越低
        # 坡度越大，可通行性越低
        traversability = 1.0 / (1.0 + height_std * 10 + slope * 5)
        
        return traversability