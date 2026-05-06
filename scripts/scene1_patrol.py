#!/usr/bin/env python3
"""
场景一：手动控制模式 — 启动仿真 + MPC 稳定，不做自主运动。

用于配合键盘遥控 (keyboard_teleop.py) 进行数据采集。

运行方式：
  python3 scripts/scene1_patrol.py              # 默认种子
  python3 scripts/scene1_patrol.py --seed 42    # 指定种子

然后另开终端：
  python3 scripts/keyboard_teleop.py             # 手动遥控
  python3 scripts/data_collector.py --output data/xxx.npz  # 采集数据
"""

import math
import os
import sys
import argparse
import threading

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
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs.point_cloud2 import read_points

    # 确保 skills 模块可导入
    script_dir = os.path.dirname(__file__)
    pkg_dir = os.path.dirname(script_dir)
    if pkg_dir not in sys.path:
        sys.path.insert(0, pkg_dir)

    from skills.controller_manager import ControllerManager

    rospy.loginfo("=" * 60)
    rospy.loginfo("  场景一：手动控制模式")
    rospy.loginfo("=" * 60)

    # ---- 传感器数据存储 ----
    sensor_lock = threading.Lock()
    latest_odom = [None]
    latest_cloud = [None]
    odom_count = [0]
    cloud_count = [0]

    def _odom_cb(msg):
        with sensor_lock:
            latest_odom[0] = msg
            odom_count[0] += 1

    def _cloud_cb(msg):
        with sensor_lock:
            latest_cloud[0] = msg
            cloud_count[0] += 1

    odom_sub = rospy.Subscriber("/odom", Odometry, _odom_cb, queue_size=10)
    cloud_sub = rospy.Subscriber("/lidar/points", PointCloud2, _cloud_cb, queue_size=10)
    rospy.loginfo("  已订阅 /odom 和 /lidar/points")

    rospy.loginfo("  等待 MPC 控制器稳定机器人姿态 (18s)...")
    rospy.sleep(18.0)

    controller = ControllerManager()
    ok = controller.use_mpc()
    if ok:
        rospy.loginfo("  MPC 控制器已就绪")
    else:
        rospy.logwarn("  MPC 切换失败，/cmd_vel 仍可被外部节点直接控制")

    cmd_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
    rospy.sleep(0.5)

    rospy.loginfo("-" * 60)
    rospy.loginfo("  仿真环境就绪，机器人保持静止。")
    rospy.loginfo("  持续发布零速 /cmd_vel，直到外部节点 (键盘/自动) 接管。")
    rospy.loginfo("  请在另一个终端运行键盘遥控或自动避障脚本：")
    rospy.loginfo("    python3 scripts/keyboard_teleop.py")
    rospy.loginfo("    python3 scripts/data_collector.py --output data/scene1_frames.npz")
    rospy.loginfo("  按 Ctrl+C 退出仿真")
    rospy.loginfo("-" * 60)

    zero_cmd = Twist()
    _last_log = rospy.Time.now()
    try:
        rate = rospy.Rate(5)
        while not rospy.is_shutdown():
            if not rospy.get_param('/keyboard_teleop/active', False):
                cmd_pub.publish(zero_cmd)

            # 定期输出传感器数据状态
            now = rospy.Time.now()
            if (now - _last_log).to_sec() >= 2.0:
                _last_log = now
                with sensor_lock:
                    odom_ok = latest_odom[0] is not None
                    cloud_ok = latest_cloud[0] is not None
                    oc = odom_count[0]
                    cc = cloud_count[0]

                odo_str = "ok" if odom_ok else "MISSING"
                lidar_str = "ok" if cloud_ok else "MISSING"

                if cloud_ok:
                    pts = []
                    for p in read_points(latest_cloud[0], field_names=("x", "y", "z"), skip_nans=True):
                        pts.append(p)
                        if len(pts) >= 50:
                            break
                    front_d = 4.5
                    for x, y, _z in pts:
                        d = math.hypot(x, y)
                        if abs(math.atan2(y, x)) < math.radians(18):
                            front_d = min(front_d, d)
                    lidar_str = f"pts~{len(pts)}+,front~{front_d:.1f}m"

                rospy.loginfo(
                    "  [sensor] odom=%s(%d)  lidar=%s(%d)",
                    odo_str, oc, lidar_str, cc,
                )

            rate.sleep()
    except KeyboardInterrupt:
        rospy.loginfo("  收到停止信号，关闭仿真...")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
