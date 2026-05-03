"""
CRAIC 2026 仿真赛技能模块
提供常用的ROS接口封装、导航工具和感知工具
"""

# 基础工具模块
from .ros_interface import ROSInterface
from .navigation_utils import NavigationUtils
from .perception_utils import PerceptionUtils

# 场景1专用模块
from .lidar_mapper import LidarMapper, Clearance
from .button_detector import ButtonDetector
from .localizer import Localizer, PoseEstimate, AnchorObservation
from .controller_manager import ControllerManager

# 比赛要求模块
from .competition_requirements import CompetitionRequirements, get_competition_requirements

__all__ = [
    # 基础工具
    'ROSInterface', 
    'NavigationUtils', 
    'PerceptionUtils',
    
    # 感知模块
    'LidarMapper',
    'Clearance',
    'ButtonDetector',
    
    # 定位模块
    'Localizer',
    'PoseEstimate',
    'AnchorObservation',
    
    # 控制模块
    'ControllerManager',
    
    # 比赛要求
    'CompetitionRequirements',
    'get_competition_requirements',
]