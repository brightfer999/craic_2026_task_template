"""
导航工具模块
提供导航相关的工具函数
"""

import numpy as np
from typing import Tuple, List, Optional
import math


class NavigationUtils:
    """导航工具类"""
    
    @staticmethod
    def calculate_distance(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点之间的距离"""
        return math.sqrt((point2[0] - point1[0])**2 + (point2[1] - point1[1])**2)
    
    @staticmethod
    def calculate_angle(point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """计算两点之间的角度（弧度）"""
        return math.atan2(point2[1] - point1[1], point2[0] - point1[0])
    
    @staticmethod
    def normalize_angle(angle: float) -> float:
        """将角度标准化到[-pi, pi]范围"""
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle
    
    @staticmethod
    def is_point_in_polygon(point: Tuple[float, float], polygon: List[Tuple[float, float]]) -> bool:
        """判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        j = n - 1
        for i in range(n):
            if ((polygon[i][1] > point[1]) != (polygon[j][1] > point[1]) and
                point[0] < (polygon[j][0] - polygon[i][0]) * (point[1] - polygon[i][1]) / 
                (polygon[j][1] - polygon[i][1]) + polygon[i][0]):
                inside = not inside
            j = i
        return inside
    
    @staticmethod
    def line_intersection(line1: Tuple[Tuple[float, float], Tuple[float, float]], 
                         line2: Tuple[Tuple[float, float], Tuple[float, float]]) -> Optional[Tuple[float, float]]:
        """计算两条线段的交点"""
        x1, y1 = line1[0]
        x2, y2 = line1[1]
        x3, y3 = line2[0]
        x4, y4 = line2[1]
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        if abs(denominator) < 1e-10:
            return None
            
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:
            x = x1 + t * (x2 - x1)
            y = y1 + t * (y2 - y1)
            return (x, y)
        return None
    
    @staticmethod
    def calculate_path_length(path: List[Tuple[float, float]]) -> float:
        """计算路径长度"""
        if len(path) < 2:
            return 0.0
        length = 0.0
        for i in range(len(path) - 1):
            length += NavigationUtils.calculate_distance(path[i], path[i+1])
        return length
    
    @staticmethod
    def interpolate_path(path: List[Tuple[float, float]], num_points: int) -> List[Tuple[float, float]]:
        """在路径上插值"""
        if len(path) < 2 or num_points < 2:
            return path
            
        total_length = NavigationUtils.calculate_path_length(path)
        segment_length = total_length / (num_points - 1)
        
        interpolated = [path[0]]
        current_length = 0.0
        current_segment = 0
        
        for i in range(1, num_points - 1):
            target_length = i * segment_length
            
            while current_length < target_length and current_segment < len(path) - 1:
                segment_start = path[current_segment]
                segment_end = path[current_segment + 1]
                segment_distance = NavigationUtils.calculate_distance(segment_start, segment_end)
                
                if current_length + segment_distance >= target_length:
                    # 在当前段内插值
                    remaining = target_length - current_length
                    ratio = remaining / segment_distance
                    x = segment_start[0] + ratio * (segment_end[0] - segment_start[0])
                    y = segment_start[1] + ratio * (segment_end[1] - segment_start[1])
                    interpolated.append((x, y))
                    break
                else:
                    current_length += segment_distance
                    current_segment += 1
            
        interpolated.append(path[-1])
        return interpolated
    
    @staticmethod
    def calculate_curvature(path: List[Tuple[float, float]]) -> List[float]:
        """计算路径的曲率"""
        if len(path) < 3:
            return [0.0] * len(path)
            
        curvatures = [0.0]  # 第一个点曲率为0
        
        for i in range(1, len(path) - 1):
            p1 = path[i-1]
            p2 = path[i]
            p3 = path[i+1]
            
            # 计算向量
            v1 = (p2[0] - p1[0], p2[1] - p1[1])
            v2 = (p3[0] - p2[0], p3[1] - p2[1])
            
            # 计算叉积
            cross_product = v1[0] * v2[1] - v1[1] * v2[0]
            
            # 计算距离
            d1 = math.sqrt(v1[0]**2 + v1[1]**2)
            d2 = math.sqrt(v2[0]**2 + v2[1]**2)
            
            if d1 * d2 > 1e-10:
                curvature = abs(cross_product) / (d1 * d2 * (d1 + d2) / 2)
            else:
                curvature = 0.0
                
            curvatures.append(curvature)
            
        curvatures.append(0.0)  # 最后一个点曲率为0
        return curvatures
    
    @staticmethod
    def smooth_path(path: List[Tuple[float, float]], weight_data: float = 0.5, 
                   weight_smooth: float = 0.1, tolerance: float = 1e-6) -> List[Tuple[float, float]]:
        """平滑路径"""
        if len(path) < 3:
            return path
            
        smoothed = list(path)
        change = tolerance
        
        while change >= tolerance:
            change = 0.0
            for i in range(1, len(path) - 1):
                for j in range(2):  # x and y
                    old_value = smoothed[i][j]
                    new_value = smoothed[i][j] + weight_data * (path[i][j] - smoothed[i][j]) + \
                               weight_smooth * (smoothed[i-1][j] + smoothed[i+1][j] - 2 * smoothed[i][j])
                    
                    if j == 0:
                        smoothed[i] = (new_value, smoothed[i][1])
                    else:
                        smoothed[i] = (smoothed[i][0], new_value)
                        
                    change += abs(old_value - new_value)
                    
        return smoothed