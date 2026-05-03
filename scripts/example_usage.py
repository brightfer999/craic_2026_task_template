#!/usr/bin/env python3
"""
Skills模块使用示例
展示如何使用新整合的skills模块完成场景1任务
"""

import rospy
from skills import LidarMapper, Localizer, ControllerManager, ButtonDetector


def main():
    """主函数：演示skills模块的使用"""
    rospy.init_node("skills_example", anonymous=True)
    
    # 初始化模块
    rospy.loginfo("初始化skills模块...")
    
    # 1. 激光雷达建图器
    lidar = LidarMapper(topic="/lidar/points", max_range=4.5)
    
    # 2. 定位器（带AprilTag锚点配置）
    apriltag_anchors = {
        1: {"name": "slope_entry", "stage_hint": "slope"},
        2: {"name": "slope_mid", "stage_hint": "slope"},
        3: {"name": "slope_exit", "stage_hint": "slope"},
        4: {"name": "stairs_entry", "stage_hint": "stairs"},
        5: {"name": "stairs_mid", "stage_hint": "stairs"},
        6: {"name": "stairs_exit", "stage_hint": "goal"},
    }
    localizer = Localizer(apriltag_anchors=apriltag_anchors)
    
    # 3. 控制器管理器
    controller = ControllerManager()
    
    # 4. 按钮检测器
    button_detector = ButtonDetector(camera_topic="/cam_h/color/image_raw/compressed")
    
    # 等待传感器数据
    rospy.loginfo("等待传感器数据...")
    rospy.sleep(2.0)
    
    # 初始化定位
    if not localizer.init_localize():
        rospy.logwarn("定位初始化失败，使用默认值")
    
    # 切换到MPC控制器
    if not controller.use_mpc():
        rospy.logwarn("MPC控制器切换失败")
    
    rospy.loginfo("开始示例控制循环...")
    
    # 控制循环
    rate = rospy.Rate(20)  # 20Hz
    start_time = rospy.Time.now()
    
    while not rospy.is_shutdown() and (rospy.Time.now() - start_time).to_sec() < 30.0:
        # 1. 获取传感器数据
        clearance = lidar.clearance()
        pose = localizer.get_pose()
        
        # 2. 避障逻辑
        if clearance.has_data:
            if clearance.front < 0.55:
                # 前方有障碍物，转向避障
                turn = lidar.preferred_turn()
                controller.send_velocity_command(
                    linear_x=0.05,
                    angular_z=turn * 0.3
                )
                rospy.loginfo(f"避障: 前方距离={clearance.front:.2f}m, 转向={turn}")
            elif clearance.front < 0.9:
                # 前方较近，减速
                controller.send_velocity_command(linear_x=0.1)
            else:
                # 前方清晰，正常前进
                controller.send_velocity_command(linear_x=0.2)
        else:
            # 无激光雷达数据，低速前进
            controller.send_velocity_command(linear_x=0.05)
        
        # 3. 检查按钮状态
        if button_detector.has_frame:
            if button_detector.is_lit("green"):
                rospy.loginfo("检测到绿色按钮亮起！")
            if button_detector.is_lit("red"):
                rospy.loginfo("检测到红色按钮亮起！")
        
        # 4. 检查AprilTag
        if localizer.has_seen_tag((1, 2, 3)):
            anchor = localizer.latest_anchor()
            rospy.loginfo(f"看到AprilTag: {anchor.name}, 距离={anchor.distance:.2f}m")
        
        # 5. 显示位姿信息
        if pose.valid:
            rospy.loginfo_throttle(
                2.0,
                f"位姿: x={pose.x:.2f}m, y={pose.y:.2f}m, yaw={pose.yaw:.2f}rad"
            )
        
        rate.sleep()
    
    # 停止机器人
    rospy.loginfo("示例结束，停止机器人")
    controller.stop_robot(1.0)
    
    # 显示统计信息
    rospy.loginfo("=== 统计信息 ===")
    rospy.loginfo(f"最终位姿: x={pose.x:.2f}m, y={pose.y:.2f}m")
    rospy.loginfo(f"定位状态: valid={pose.valid}")
    rospy.loginfo(f"激光雷达数据: available={lidar.has_data}")


if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"示例运行失败: {e}")
        import traceback
        traceback.print_exc()