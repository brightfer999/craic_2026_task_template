#!/usr/bin/env python3
"""
Skills模块测试脚本
测试所有skills模块的基本功能
"""

import sys
import os

# 添加skills目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试所有模块的导入"""
    print("测试模块导入...")
    
    try:
        from skills import LidarMapper, Clearance
        print("✓ LidarMapper导入成功")
    except Exception as e:
        print(f"✗ LidarMapper导入失败: {e}")
        return False
    
    try:
        from skills import ButtonDetector
        print("✓ ButtonDetector导入成功")
    except Exception as e:
        print(f"✗ ButtonDetector导入失败: {e}")
        return False
    
    try:
        from skills import Localizer, PoseEstimate, AnchorObservation
        print("✓ Localizer导入成功")
    except Exception as e:
        print(f"✗ Localizer导入失败: {e}")
        return False
    
    try:
        from skills import ControllerManager
        print("✓ ControllerManager导入成功")
    except Exception as e:
        print(f"✗ ControllerManager导入失败: {e}")
        return False
    
    try:
        from skills import NavigationUtils
        print("✓ NavigationUtils导入成功")
    except Exception as e:
        print(f"✗ NavigationUtils导入失败: {e}")
        return False
    
    try:
        from skills import PerceptionUtils
        print("✓ PerceptionUtils导入成功")
    except Exception as e:
        print(f"✗ PerceptionUtils导入失败: {e}")
        return False
    
    try:
        from skills import ROSInterface
        print("✓ ROSInterface导入成功")
    except Exception as e:
        print(f"✗ ROSInterface导入失败: {e}")
        return False
    
    return True


def test_navigation_utils():
    """测试NavigationUtils功能"""
    print("\n测试NavigationUtils...")
    
    from skills import NavigationUtils
    import math
    
    # 测试距离计算
    dist = NavigationUtils.calculate_distance((0, 0), (3, 4))
    assert abs(dist - 5.0) < 1e-6, f"距离计算错误: {dist}"
    print("✓ 距离计算正确")
    
    # 测试角度计算
    angle = NavigationUtils.calculate_angle((0, 0), (1, 1))
    assert abs(angle - math.pi/4) < 1e-6, f"角度计算错误: {angle}"
    print("✓ 角度计算正确")
    
    # 测试角度标准化
    angle = NavigationUtils.normalize_angle(3 * math.pi)
    assert abs(angle - math.pi) < 1e-6, f"角度标准化错误: {angle}"
    print("✓ 角度标准化正确")
    
    # 测试点是否在多边形内
    polygon = [(0, 0), (1, 0), (1, 1), (0, 1)]
    assert NavigationUtils.is_point_in_polygon((0.5, 0.5), polygon), "点应该在多边形内"
    assert not NavigationUtils.is_point_in_polygon((1.5, 0.5), polygon), "点不应该在多边形内"
    print("✓ 多边形包含测试正确")
    
    # 测试路径长度计算
    path = [(0, 0), (1, 0), (1, 1)]
    length = NavigationUtils.calculate_path_length(path)
    assert abs(length - 2.0) < 1e-6, f"路径长度计算错误: {length}"
    print("✓ 路径长度计算正确")
    
    print("NavigationUtils测试通过！")
    return True


def test_perception_utils():
    """测试PerceptionUtils功能"""
    print("\n测试PerceptionUtils...")
    
    from skills import PerceptionUtils
    import numpy as np
    
    # 创建实例
    perception = PerceptionUtils()
    
    # 测试点云过滤
    points = np.array([
        [0, 0, 0],
        [1, 1, 1],
        [5, 5, 5],
        [-1, -1, -1]
    ])
    
    filtered = perception.filter_point_cloud(
        points,
        x_range=(-2, 2),
        y_range=(-2, 2),
        z_range=(-2, 2)
    )
    
    assert len(filtered) == 3, f"过滤后点数错误: {len(filtered)}"
    print("✓ 点云过滤正确")
    
    # 测试点云下采样
    if len(points) > 0:
        downsampled = perception.downsample_point_cloud(points, voxel_size=1.0)
        print(f"✓ 点云下采样完成: {len(points)} -> {len(downsampled)} 点")
    
    print("PerceptionUtils测试通过！")
    return True


def test_data_classes():
    """测试数据类"""
    print("\n测试数据类...")
    
    from skills import Clearance, PoseEstimate, AnchorObservation
    
    # 测试Clearance数据类
    clearance = Clearance(
        front=1.0,
        left_front=1.5,
        right_front=1.2,
        left=2.0,
        right=1.8,
        rear=3.0,
        has_data=True
    )
    assert clearance.front == 1.0
    assert clearance.has_data == True
    print("✓ Clearance数据类正确")
    
    # 测试PoseEstimate数据类（不使用rospy.Time）
    from unittest.mock import MagicMock
    mock_stamp = MagicMock()
    
    pose = PoseEstimate(
        x=1.0,
        y=2.0,
        yaw=0.5,
        stamp=mock_stamp,
        valid=True
    )
    assert pose.x == 1.0
    assert pose.valid == True
    print("✓ PoseEstimate数据类正确")
    
    # 测试AnchorObservation数据类
    anchor = AnchorObservation(
        tag_id=1,
        name="test_tag",
        stage_hint="test_stage",
        forward=1.0,
        lateral=0.5,
        distance=1.12,
        stamp=mock_stamp
    )
    assert anchor.tag_id == 1
    assert anchor.name == "test_tag"
    print("✓ AnchorObservation数据类正确")
    
    print("数据类测试通过！")
    return True


def main():
    """主测试函数"""
    print("=== Skills模块测试 ===\n")
    
    tests = [
        ("模块导入", test_imports),
        ("NavigationUtils", test_navigation_utils),
        ("PerceptionUtils", test_perception_utils),
        ("数据类", test_data_classes),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
            else:
                failed += 1
                print(f"✗ {test_name}测试失败")
        except Exception as e:
            failed += 1
            print(f"✗ {test_name}测试异常: {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n=== 测试结果 ===")
    print(f"通过: {passed}")
    print(f"失败: {failed}")
    print(f"总计: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print(f"\n❌ {failed}个测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())