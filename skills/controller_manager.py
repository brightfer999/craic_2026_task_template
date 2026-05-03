"""
控制器管理模块
提供人形机器人控制器的切换和管理功能
"""

import rospy
from geometry_msgs.msg import Twist
from typing import Optional, Set


class ControllerManager:
    """人形机器人控制器管理器
    
    提供：
    - MPC控制器（平地行走）
    - AMP控制器（复杂地形）
    - 控制器切换服务
    - 机器人停止控制
    """
    
    # 控制器名称常量
    MPC = "mpc"
    AMP = "amp_controller"
    LEJU_WALK = "leju_walk_controller"
    
    def __init__(self, service_name: str = "/humanoid_controller/switch_controller"):
        """初始化控制器管理器
        
        Args:
            service_name: 控制器切换服务名称
        """
        self._service_name = service_name
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._current: Optional[str] = None
        self._switch = None
        self._available_controllers: Optional[Set[str]] = None
        
    def use_mpc(self) -> bool:
        """切换到MPC控制器
        
        Returns:
            是否切换成功
        """
        return self.use(self.MPC)
        
    def use_amp(self) -> bool:
        """切换到AMP控制器
        
        如果AMP不可用，会自动降级到leju_walk_controller
        
        Returns:
            是否切换成功
        """
        if self._available_controllers is None:
            self._probe_controllers()
            
        if self.AMP in self._available_controllers:
            return self.use(self.AMP)
            
        rospy.loginfo("amp_controller not available, using leju_walk_controller")
        return self.use(self.LEJU_WALK)
        
    def use(self, controller_name: str, timeout: float = 5.0, settle_time: float = 1.0) -> bool:
        """切换到指定控制器
        
        Args:
            controller_name: 控制器名称
            timeout: 等待服务超时时间（秒）
            settle_time: 切换后稳定时间（秒）
            
        Returns:
            是否切换成功
        """
        if self._current == controller_name:
            return True
            
        # 先停止机器人
        self.stop_robot(2.0)
        
        # 导入服务类型
        try:
            from kuavo_msgs.srv import switchController
        except ImportError:
            rospy.logwarn("kuavo_msgs not available, controller switch disabled")
            return False
            
        # 等待服务可用
        if self._switch is None:
            try:
                rospy.wait_for_service(self._service_name, timeout=timeout)
                self._switch = rospy.ServiceProxy(self._service_name, switchController)
            except rospy.ROSException as e:
                rospy.logwarn(f"Controller service not available: {e}")
                return False
                
        # 尝试切换控制器
        for attempt in range(5):
            try:
                response = self._switch(controller_name)
            except Exception as exc:
                rospy.logwarn(f"Controller switch to {controller_name} failed (attempt {attempt + 1}): {exc}")
                rospy.sleep(1.0)
                continue
                
            if getattr(response, "success", False):
                self._current = controller_name
                rospy.sleep(settle_time)
                rospy.loginfo(f"Using controller: {controller_name}")
                return True
                
            rospy.logwarn(
                f"Controller switch to {controller_name} rejected (attempt {attempt + 1}): "
                f"{getattr(response, 'message', '')}"
            )
            self.stop_robot(2.0)
            
        return False
        
    def stop_robot(self, duration: float = 0.2):
        """停止机器人
        
        Args:
            duration: 停止命令持续时间（秒）
        """
        cmd = Twist()
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(20)
        
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self._cmd_pub.publish(cmd)
            rate.sleep()
            
    def send_velocity_command(self, linear_x: float = 0.0, linear_y: float = 0.0, angular_z: float = 0.0):
        """发送速度命令
        
        Args:
            linear_x: 前进速度（m/s）
            linear_y: 侧移速度（m/s）
            angular_z: 转向角速度（rad/s）
        """
        cmd = Twist()
        cmd.linear.x = linear_x
        cmd.linear.y = linear_y
        cmd.angular.z = angular_z
        self._cmd_pub.publish(cmd)
        
    @property
    def current_controller(self) -> Optional[str]:
        """获取当前控制器名称"""
        return self._current
        
    @property
    def is_mpc(self) -> bool:
        """检查是否使用MPC控制器"""
        return self._current == self.MPC
        
    @property
    def is_amp(self) -> bool:
        """检查是否使用AMP控制器"""
        return self._current == self.AMP
        
    def get_available_controllers(self) -> Set[str]:
        """获取可用的控制器列表
        
        Returns:
            可用控制器名称集合
        """
        if self._available_controllers is None:
            self._probe_controllers()
        return self._available_controllers.copy()
        
    def _probe_controllers(self):
        """探测可用的控制器"""
        try:
            from kuavo_msgs.srv import switchController
            
            rospy.wait_for_service(self._service_name, timeout=5.0)
            switch_svc = rospy.ServiceProxy(self._service_name, switchController)
            
            # 尝试切换到AMP来测试是否可用
            resp = switch_svc(self.AMP)
            if getattr(resp, "success", False):
                self._available_controllers = {self.MPC, self.AMP, self.LEJU_WALK}
                # 切换回MPC
                switch_svc(self.MPC)
            else:
                self._available_controllers = {self.MPC, self.LEJU_WALK}
                
        except Exception as e:
            rospy.logwarn(f"Failed to probe controllers: {e}")
            self._available_controllers = {self.MPC, self.LEJU_WALK}
            
    def reset(self):
        """重置控制器管理器状态"""
        self._current = None
        self._switch = None
        self._available_controllers = None