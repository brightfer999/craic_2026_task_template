#!/usr/bin/env python3.8
"""
自动避障演示 — 基于官方 LidarMapper + ControllerManager 模板的完整感知-控制闭环。

运行方式:
  python3 scripts/auto_avoidance_demo.py

依赖 (需先启动场景仿真):
  - /lidar/points  (PointCloud2)   激光雷达点云
  - /odom          (Odometry)      里程计
  - /cmd_vel       (Twist)         速度控制输出
  - /humanoid_controller/switch_controller  控制器切换

工作流程:
  1. 等待 MPC 控制器就绪 (15s 稳定时间)
  2. 初始化 ObstacleDetector + ControllerManager + AvoidanceController
  3. 20Hz 循环: 感知 → 避障决策 → 发布 /cmd_vel
  4. Ctrl+C 安全停止
"""

import os
import sys
import signal
import time

import rospy
from geometry_msgs.msg import Twist

# 确保 skills 模块可导入
_script_dir = os.path.dirname(os.path.abspath(__file__))
_pkg_dir = os.path.dirname(_script_dir)
if _pkg_dir not in sys.path:
    sys.path.insert(0, _pkg_dir)

from skills.perception_avoidance import ObstacleDetector, AvoidanceController
from skills.controller_manager import ControllerManager


class AutoAvoidanceDemo:
    """自动避障演示节点。"""

    def __init__(self):
        rospy.init_node("auto_avoidance_demo", anonymous=True)
        self._running = True
        signal.signal(signal.SIGINT, self._signal_handler)

        rospy.loginfo("=" * 60)
        rospy.loginfo("  自动避障演示 — 感知-控制闭环启动")
        rospy.loginfo("=" * 60)

        # 等待 MPC 稳定
        rospy.loginfo("  等待 MPC 控制器就绪 (15s)...")
        rospy.sleep(15.0)

        # 初始化组件
        self._controller = ControllerManager()
        ok = self._controller.use_mpc()
        if not ok:
            rospy.logwarn("  MPC 切换失败，尝试直接发布 /cmd_vel")
        else:
            rospy.loginfo("  MPC 控制器就绪")

        self._detector = ObstacleDetector()
        self._avoidance = AvoidanceController(
            detector=self._detector,
            controller=self._controller,
        )

        # 独立 cmd_vel 发布器 (兜底)
        self._cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)

        # 等待激光雷达数据
        rospy.loginfo("  等待激光雷达数据...")
        deadline = time.time() + 6.0
        while not rospy.is_shutdown() and time.time() < deadline:
            if self._detector.has_data:
                rospy.loginfo("  激光雷达数据就绪")
                break
            rospy.sleep(0.2)
        else:
            rospy.logwarn("  激光雷达数据等待超时")

    def _signal_handler(self, sig, frame):
        rospy.loginfo("\n  收到停止信号，正在安全停止...")
        self._running = False

    def run(self):
        rate = rospy.Rate(20)
        last_has_data = False
        no_data_timer = 0.0

        rospy.loginfo("  开始自动避障控制循环 (20Hz)")
        rospy.loginfo("  按 Ctrl+C 停止")
        rospy.loginfo("-" * 60)

        while self._running and not rospy.is_shutdown():
            if self._detector.has_data:
                if not last_has_data:
                    rospy.loginfo("  激光雷达数据恢复")
                    no_data_timer = 0.0
                last_has_data = True

                cmd = self._avoidance.step()
                self._cmd_pub.publish(cmd)

            else:
                if last_has_data:
                    rospy.logwarn("  激光雷达数据丢失，停止等待...")
                    last_has_data = False
                no_data_timer += 0.05

                if no_data_timer > 2.0:
                    # 超时停止
                    cmd = Twist()
                    self._cmd_pub.publish(cmd)

                rate.sleep()
                continue

            rate.sleep()

        # 安全停止
        self._avoidance.stop()
        rospy.loginfo("  已安全停止，退出。")


def main():
    try:
        AutoAvoidanceDemo().run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
