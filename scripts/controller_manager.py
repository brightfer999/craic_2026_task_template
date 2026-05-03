#!/usr/bin/env python3

import rospy
from geometry_msgs.msg import Twist


class ControllerManager:
    """Small wrapper around the humanoid controller switching service."""

    MPC = "mpc"
    AMP = "amp_controller"
    LEJU_WALK = "leju_walk_controller"

    def __init__(self, service_name="/humanoid_controller/switch_controller"):
        self._service_name = service_name
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._current = None
        self._switch = None
        self._available_controllers = None

    def use_mpc(self) -> bool:
        return self.use(self.MPC)

    def use_amp(self) -> bool:
        if self._available_controllers is None:
            self._probe_controllers()
        if self.AMP in self._available_controllers:
            return self.use(self.AMP)
        rospy.loginfo("amp_controller not available, using leju_walk_controller")
        return self.use(self.LEJU_WALK)

    def use(self, controller_name: str, timeout: float = 5.0, settle_time: float = 1.0) -> bool:
        if self._current == controller_name:
            return True

        self.stop_robot(2.0)

        from kuavo_msgs.srv import switchController

        if self._switch is None:
            rospy.wait_for_service(self._service_name, timeout=timeout)
            self._switch = rospy.ServiceProxy(self._service_name, switchController)

        for attempt in range(5):
            try:
                response = self._switch(controller_name)
            except Exception as exc:
                rospy.logwarn("Controller switch to %s failed (attempt %d): %s", controller_name, attempt + 1, exc)
                rospy.sleep(1.0)
                continue

            if getattr(response, "success", False):
                self._current = controller_name
                rospy.sleep(settle_time)
                rospy.loginfo("Using controller: %s", controller_name)
                return True

            rospy.logwarn("Controller switch to %s rejected (attempt %d): %s", controller_name, attempt + 1, getattr(response, "message", ""))
            self.stop_robot(2.0)

        return False

    def stop_robot(self, duration: float = 0.2):
        cmd = Twist()
        end_time = rospy.Time.now() + rospy.Duration(duration)
        rate = rospy.Rate(20)
        while not rospy.is_shutdown() and rospy.Time.now() < end_time:
            self._cmd_pub.publish(cmd)
            rate.sleep()

    def _probe_controllers(self):
        from kuavo_msgs.srv import switchController
        try:
            rospy.wait_for_service(self._service_name, timeout=5.0)
            switch_svc = rospy.ServiceProxy(self._service_name, switchController)
            resp = switch_svc(self.AMP)
            if getattr(resp, "success", False):
                self._available_controllers = {self.MPC, self.AMP, self.LEJU_WALK}
                switch_svc(self.MPC)
            else:
                self._available_controllers = {self.MPC, self.LEJU_WALK}
        except Exception:
            self._available_controllers = {self.MPC, self.LEJU_WALK}
