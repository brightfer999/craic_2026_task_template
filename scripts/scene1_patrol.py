#!/usr/bin/env python3
"""
Scene 1 entrypoint.

Default mode runs the seed0-focused autonomous obstacle-zone traversal:
  python3 src/craic_task_template/scripts/scene1_patrol.py --seed 0

Manual mode keeps the original simulator + zero-velocity behavior for keyboard
teleop and data collection:
  python3 src/craic_task_template/scripts/scene1_patrol.py --seed 0 --manual
"""

import argparse
import json
import math
import os
import sys
import threading

SIM_UTILS_DIR = os.path.join(os.path.dirname(__file__), "../../craic_simulator/utils")


def _prepare_import_paths():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    pkg_dir = os.path.dirname(script_dir)
    for path in (script_dir, pkg_dir):
        if path not in sys.path:
            sys.path.insert(0, path)


def _start_scene(seed: int, node_name: str):
    if SIM_UTILS_DIR not in sys.path:
        sys.path.insert(0, SIM_UTILS_DIR)
    from sim_launcher import SimLauncher

    launcher = SimLauncher(scene="scene1", seed=seed)
    launcher.start(node_name=node_name)
    return launcher


def _load_ros_params_file(path: str) -> int:
    if not path:
        return 0
    import rospy

    try:
        with open(path, "r", encoding="utf-8") as fh:
            params = json.load(fh)
    except Exception as exc:
        rospy.logwarn("  参数文件读取失败：%s (%s)", path, exc)
        return 0

    count = 0
    for name, value in params.items():
        if not isinstance(name, str):
            continue
        if name.startswith("_"):
            name = name[1:]
        rospy.set_param("~" + name, value)
        count += 1
    rospy.loginfo("  已加载自迭代参数：%s (%d 项)", path, count)
    return count


def run_manual_mode(args) -> int:
    _start_scene(args.seed, "scene1_patrol")

    import rospy
    from geometry_msgs.msg import Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import PointCloud2
    from sensor_msgs.point_cloud2 import read_points

    _prepare_import_paths()
    from skills.controller_manager import ControllerManager

    rospy.loginfo("=" * 60)
    rospy.loginfo("  场景一：手动控制模式")
    rospy.loginfo("=" * 60)

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

    rospy.Subscriber("/odom", Odometry, _odom_cb, queue_size=10)
    rospy.Subscriber("/lidar/points", PointCloud2, _cloud_cb, queue_size=10)
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
    rospy.loginfo("  可另开终端运行：")
    rospy.loginfo("    python3 scripts/keyboard_teleop.py")
    rospy.loginfo("    python3 scripts/data_collector.py --output data/scene1_frames.npz")
    rospy.loginfo("  按 Ctrl+C 退出仿真")
    rospy.loginfo("-" * 60)

    zero_cmd = Twist()
    last_log = rospy.Time.now()
    rate = rospy.Rate(5)
    while not rospy.is_shutdown():
        if not rospy.get_param("/keyboard_teleop/active", False):
            cmd_pub.publish(zero_cmd)

        now = rospy.Time.now()
        if (now - last_log).to_sec() >= 2.0:
            last_log = now
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
                lidar_str = "pts~%d+,front~%.1fm" % (len(pts), front_d)

            rospy.loginfo("  [sensor] odom=%s(%d)  lidar=%s(%d)", odo_str, oc, lidar_str, cc)

        rate.sleep()

    return 0


def run_auto_obstacle_mode(args) -> int:
    _start_scene(args.seed, "scene1_patrol")

    import rospy

    _prepare_import_paths()
    from controller_manager import ControllerManager
    from localization import Localizer
    from terrain_traverser import TerrainTraverser
    from waypoints import SEED0_OBSTACLE_STAGES

    rospy.loginfo("=" * 60)
    rospy.loginfo("  场景一：自动避障穿越障碍区 (seed=%d)", args.seed)
    rospy.loginfo("=" * 60)
    rospy.loginfo("  规则约束：不使用 /ground_truth/state，不使用世界坐标路线")

    settle = float(rospy.get_param("~startup_settle_time", args.settle_time))
    rospy.loginfo("  等待机器人和控制器稳定 (%.1fs)...", settle)
    rospy.sleep(settle)

    controller = ControllerManager()
    if controller.use_mpc():
        rospy.loginfo("  MPC 控制器已就绪")
    else:
        rospy.logwarn("  MPC 切换失败，将继续使用 /cmd_vel 低速安全指令")

    localizer = Localizer()
    odom_ok = localizer.init_localize(wait_timeout=4.0)
    rospy.loginfo("  /odom 状态：%s", "ready" if odom_ok else "missing")

    _load_ros_params_file(args.params_file)

    if args.planner_log:
        rospy.set_param("~planner_log_path", args.planner_log)
        rospy.loginfo("  planner 日志：%s", args.planner_log)

    traverser = TerrainTraverser(localizer, controller)
    lidar_ok = traverser.wait_for_lidar(timeout=6.0)
    rospy.loginfo("  /lidar/points 状态：%s", "ready" if lidar_ok else "missing")
    traverser.log_obstacle_status(prefix="startup")

    if args.startup_scan:
        traverser.perform_startup_lidar_rotation(duration=args.scan_duration, angular_z=args.scan_speed)
        traverser.log_obstacle_status(prefix="after_scan")

    direction_ok = traverser.initial_direction_check(timeout=4.0)
    if not direction_ok:
        rospy.logwarn("  开局方向判据不确定或疑似反向；障碍区模式不执行大范围掉头，继续低速 gap-follow")

    leave_stage, obstacle_stage = SEED0_OBSTACLE_STAGES
    rospy.loginfo("  执行阶段：%s - %s", leave_stage.name, leave_stage.description)
    rospy.loginfo("  执行阶段：%s - %s", obstacle_stage.name, obstacle_stage.description)
    completed = traverser.cross_seed0_obstacle_zone(leave_stage, obstacle_stage)

    traverser.stop(1.0)

    travelled = localizer.distance_since_stage_reset()
    rospy.loginfo("-" * 60)
    rospy.loginfo("  自动避障结果：%s", "completed" if completed else "failed_or_partial")
    rospy.loginfo("  最后阶段相对行程：%.2fm", travelled)
    rospy.loginfo("  障碍近距计数：%d", traverser.obstacle_hit_count)
    rospy.loginfo("  stop-after=%s，机器人已安全停车", args.stop_after)
    rospy.loginfo("-" * 60)

    if args.stop_after == "none":
        rospy.loginfo("  完整场景一未在本次范围内实现；当前版本仅保证障碍区优先。")

    return 0 if completed else 2


def parse_args():
    parser = argparse.ArgumentParser(description="场景一：默认自动避障穿越障碍区")
    parser.add_argument("--seed", type=int, default=0, help="随机种子（默认0）")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--auto", action="store_true", help="自动避障模式（默认）")
    mode.add_argument("--manual", action="store_true", help="手动/数据采集模式")
    parser.add_argument(
        "--stop-after",
        choices=("obstacle", "none"),
        default="obstacle",
        help="自动模式停止点；none 仅预留后续完整场景一",
    )
    parser.add_argument("--settle-time", type=float, default=18.0, help="启动后稳定等待时间")
    parser.add_argument("--startup-scan", action="store_true", help="穿越前进行原地低速雷达扫描")
    parser.add_argument("--scan-duration", type=float, default=4.0, help="启动扫描时长")
    parser.add_argument("--scan-speed", type=float, default=-0.25, help="启动扫描角速度")
    parser.add_argument("--planner-log", default="", help="自动模式每帧规划 JSONL 日志路径")
    parser.add_argument("--params-file", default="", help="加载 auto_optimize_obstacle.py 生成的 best_params.json")
    args, _ros_args = parser.parse_known_args()
    return args


def main():
    args = parse_args()
    if args.manual:
        return run_manual_mode(args)
    return run_auto_obstacle_mode(args)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        pass
