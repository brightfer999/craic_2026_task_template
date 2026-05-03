#!/usr/bin/env python3
"""
测试所有skills模块的导入和基本功能
"""

import sys
import os

# 添加skills目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def test_imports():
    """测试所有模块的导入"""
    print("测试所有skills模块导入...\n")
    
    try:
        from skills import (
            ROSInterface, NavigationUtils, PerceptionUtils,
            LidarMapper, Clearance, ButtonDetector,
            Localizer, PoseEstimate, AnchorObservation,
            ControllerManager, CompetitionRequirements, get_competition_requirements
        )
        
        modules = [
            ("ROSInterface", "基础工具模块", ROSInterface),
            ("NavigationUtils", "基础工具模块", NavigationUtils),
            ("PerceptionUtils", "基础工具模块", PerceptionUtils),
            ("LidarMapper", "感知模块", LidarMapper),
            ("Clearance", "感知模块", Clearance),
            ("ButtonDetector", "感知模块", ButtonDetector),
            ("Localizer", "定位模块", Localizer),
            ("PoseEstimate", "定位模块", PoseEstimate),
            ("AnchorObservation", "定位模块", AnchorObservation),
            ("ControllerManager", "控制模块", ControllerManager),
            ("CompetitionRequirements", "比赛要求模块", CompetitionRequirements),
            ("get_competition_requirements", "比赛要求模块", get_competition_requirements),
        ]
        
        success_count = 0
        fail_count = 0
        
        for module_name, category, module in modules:
            try:
                print(f"✓ {module_name} ({category}) - 导入成功")
                success_count += 1
            except Exception as e:
                print(f"✗ {module_name} ({category}) - 导入失败: {e}")
                fail_count += 1
        
        print(f"\n导入测试结果: {success_count} 成功, {fail_count} 失败")
        return fail_count == 0
        
    except Exception as e:
        print(f"✗ 导入skills模块失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_competition_requirements():
    """测试比赛要求模块功能"""
    print("\n测试比赛要求模块功能...\n")
    
    try:
        from skills import CompetitionRequirements
        
        # 创建实例
        requirements = CompetitionRequirements()
        
        # 测试基本功能
        sensors = requirements.get_allowed_sensors()
        print(f"✓ 获取允许传感器: {len(sensors)} 个")
        
        forbidden = requirements.get_forbidden_interfaces()
        print(f"✓ 获取禁止接口: {len(forbidden)} 个")
        
        required_files = requirements.get_required_files()
        print(f"✓ 获取必须文件: {len(required_files)} 个")
        
        forbidden_actions = requirements.get_forbidden_actions()
        print(f"✓ 获取严禁事项: {len(forbidden_actions)} 个")
        
        general_reqs = requirements.get_general_requirements()
        print(f"✓ 获取通用要求: {len(general_reqs)} 个")
        
        docker_reqs = requirements.get_docker_requirements()
        print(f"✓ 获取Docker要求: {len(docker_reqs)} 个")
        
        # 测试验证功能
        import tempfile
        with tempfile.TemporaryDirectory() as temp_dir:
            # 创建测试目录结构
            os.makedirs(os.path.join(temp_dir, "craic_task_template", "scripts"))
            
            # 创建测试文件
            with open(os.path.join(temp_dir, "craic_task_template", "scripts", "scene1_patrol.py"), "w") as f:
                f.write("# Test")
            
            with open(os.path.join(temp_dir, "craic_task_template", "scripts", "scene2_sorting.py"), "w") as f:
                f.write("# Test")
            
            # 验证提交
            results = requirements.validate_submission(temp_dir)
            print(f"✓ 验证功能正常: {len(results)} 项检查")
        
        print("\n比赛要求模块功能测试通过！")
        return True
        
    except Exception as e:
        print(f"✗ 比赛要求模块功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主测试函数"""
    print("=== Skills模块综合测试 ===\n")
    
    # 测试导入
    import_success = test_imports()
    
    # 测试比赛要求模块功能
    if import_success:
        requirements_success = test_competition_requirements()
    else:
        requirements_success = False
    
    # 总结
    print("\n=== 测试总结 ===")
    print(f"模块导入: {'✓ 通过' if import_success else '✗ 失败'}")
    print(f"功能测试: {'✓ 通过' if requirements_success else '✗ 失败'}")
    
    if import_success and requirements_success:
        print("\n🎉 所有测试通过！")
        return 0
    else:
        print("\n❌ 部分测试失败")
        return 1


if __name__ == "__main__":
    sys.exit(main())