#!/usr/bin/env python3
"""
场景一：手动控制模式 — 仅启动仿真 + MPC 稳定，不做任何自主运动。

用于配合键盘遥控 (keyboard_teleop.py) 进行数据采集。

运行方式：
  python3 scripts/scene1_patrol.py              # 默认种子
  python3 scripts/scene1_patrol.py --seed 42    # 指定种子

然后另开终端：
  python3 scripts/keyboard_teleop.py             # 手动遥控
  python3 scripts/data_collector.py --output data/xxx.npz  # 采集数据
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from sim_launcher import SimLauncher


def main():
    parser = argparse.ArgumentParser(description="场景一：手动控制（仅启动仿真，不自主运动）")
    parser.add_argument('--seed', type=int, default=0, help='随机种子（默认0）')
    args = parser.parse_args()

    launcher = SimLauncher(scene="scene1", seed=args.seed)
    launcher.start(node_name="scene1_patrol")

    import rospy
    from geometry_msgs.msg import Twist

    # 确保 skills 模块可导入
    script_dir = os.path.dirname(__file__)
    pkg_dir = os.path.dirname(script_dir)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from skills.controller_manager import ControllerManager

    rospy.loginfo("=" * 60)
    rospy.loginfo("  场景一：手动控制模式")
    rospy.loginfo("=" * 60)

    rospy.loginfo("  等待 MPC 控制器稳定机器人姿态 (18s)...")
    rospy.sleep(18.0)

    controller = ControllerManager()
    ok = controller.use_mpc()
    if ok:
        rospy.loginfo("  MPC 控制器已就绪")
    else:
        rospy.logwarn("  MPC 切换失败，/cmd_vel 仍可被外部节点直接控制")

    # 创建 cmd_vel publisher，持续发布零速以保持机器人静止
    # 这确保即使 MPC 切换失败，机器人也不会自主运动
    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.sleep(0.5)  # 等待 publisher 注册到 master

    rospy.loginfo("-" * 60)
    rospy.loginfo("  仿真环境就绪，机器人保持静止。")
    rospy.loginfo("  持续发布零速 /cmd_vel，直到外部节点 (键盘/自动) 接管。")
    rospy.loginfo("  请在另一个终端运行键盘遥控或自动避障脚本：")
    rospy.loginfo("    python3 scripts/keyboard_teleop.py")
    rospy.loginfo("    python3 scripts/data_collector.py --output data/scene1_frames.npz")
    rospy.loginfo("  按 Ctrl+C 退出仿真")
    rospy.loginfo("-" * 60)

    zero_cmd = Twist()
    try:
        rate = rospy.Rate(5)  # 低频安全兜底
        while not rospy.is_shutdown():
            # 仅在键盘遥控未接管时发布零速，避免与 keyboard_teleop 竞争
            if not rospy.get_param('/keyboard_teleop/active', False):
                cmd_pub.publish(zero_cmd)
            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("  收到停止信号，关闭仿真...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
