#!/usr/bin/env python3
"""
场景一：安全巡检任务脚本

路线：起点 → 障碍区 → 操作台1 → 复杂地形区 → 操作台2 → 斜坡区 → 台阶区 → 终点
附加任务（可选）：终点 → 台阶区 → 斜坡区 → 复杂地形区 → 障碍区 → 起点

运行方式：
  python3 scene1_patrol.py              # 使用默认种子
  python3 scene1_patrol.py --seed 123   # 指定随机种子

可用接口：
  - /cmd_vel (geometry_msgs/Twist)               发送速度指令: linear.x=前进, linear.y=侧移, angular.z=转向
  - /kuavo_arm_traj (sensor_msgs/JointState)      手臂轨迹控制
  - /lidar/points (sensor_msgs/PointCloud2)       雷达点云数据（局部实时建图/避障）
  - /odom (nav_msgs/Odometry)                     阶段内相对里程计
  - /tag_detections (apriltag_ros)                可选 AprilTag 语义锚点
  - /sensors_data_raw (kuavo_msgs/sensorsData)    传感器原始数据（IMU、关节等）
  - /humanoid_controller/switch_controller        切换控制器（mpc/amp_controller）
  - GripperController (craic_simulator)           夹爪控制（用于按钮操作）
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../craic_simulator/utils'))
from sim_launcher import SimLauncher


def main():
    parser = argparse.ArgumentParser(description="场景一：安全巡检")
    parser.add_argument('--seed', type=int, default=0, help='随机种子（控制机器人初始位姿，默认0）')
    args = parser.parse_args()

    launcher = SimLauncher(scene="scene1", seed=args.seed)
    launcher.start(node_name="scene1_patrol")

    import rospy

    script_dir = os.path.dirname(__file__)
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    from controller_manager import ControllerManager
    from localization import Localizer
    from state_machine import PatrolFSM

    rospy.loginfo("=== 场景一：安全巡检任务启动 ===")
    rospy.loginfo("等待 MPC 控制器稳定机器人姿态...")
    rospy.sleep(8.0)

    controller = ControllerManager()
    controller.use_mpc()

    localizer = Localizer()
    ok = localizer.init_localize()
    if not ok:
        rospy.logwarn("相对里程计暂不可用，仍按雷达局部避障进行低速试探")

    pose = localizer.get_pose()
    rospy.loginfo(
        "初始相对位姿: dx=%.2f dy=%.2f dyaw=%.2f (odom_valid=%s, anchor_valid=%s)",
        pose.x,
        pose.y,
        pose.yaw,
        pose.valid,
        localizer.anchor_is_valid,
    )

    patrol = PatrolFSM(localizer, controller)
    result = patrol.run()
    rospy.loginfo(
        "场景一主任务完成=%s，用时=%.1fs，失败阶段=%s",
        result.completed,
        result.elapsed_time,
        result.failed_states,
    )

    if patrol.should_attempt_extra(result):
        rospy.loginfo("满足附加任务条件，开始反向巡检")
        extra_result = patrol.run_reverse()
        rospy.loginfo(
            "场景一附加任务完成=%s，用时=%.1fs，失败阶段=%s",
            extra_result.completed,
            extra_result.elapsed_time,
            extra_result.failed_states,
        )
    else:
        rospy.loginfo("不满足附加任务条件，停在终点")

    controller.stop_robot(1.0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        pass
