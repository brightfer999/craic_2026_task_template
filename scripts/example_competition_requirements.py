#!/usr/bin/env python3
"""
比赛要求模块使用示例
展示如何使用CompetitionRequirements模块获取比赛规则和验证提交
"""

import sys
import os

# 添加skills目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from skills import CompetitionRequirements


def main():
    """主函数：演示比赛要求模块的使用"""
    print("=== CRAIC 2026 比赛要求模块使用示例 ===\n")
    
    # 1. 创建比赛要求实例
    requirements = CompetitionRequirements()
    
    # 2. 打印比赛要求摘要
    print("1. 比赛要求摘要：")
    requirements.print_summary()
    
    # 3. 获取具体要求
    print("\n2. 获取具体要求：")
    
    # 获取允许使用的传感器
    sensors = requirements.get_allowed_sensors()
    print(f"允许使用的传感器: {sensors}")
    
    # 获取禁止使用的接口
    forbidden = requirements.get_forbidden_interfaces()
    print(f"禁止使用的接口: {forbidden}")
    
    # 获取允许修改范围
    scope = requirements.get_allowed_modification_scope()
    print(f"允许修改范围: {scope}")
    
    # 获取导航原则
    principles = requirements.get_navigation_principles()
    print(f"导航原则: {principles}")
    
    # 4. 获取文件结构要求
    print("\n3. 文件结构要求：")
    
    # 情况一：使用仿真环境自带控制器
    structure1 = requirements.get_file_structure("情况一：使用仿真环境自带控制器")
    print("情况一：使用仿真环境自带控制器")
    for line in structure1:
        print(f"  {line}")
    
    # 情况二：自行修改控制器
    structure2 = requirements.get_file_structure("情况二：自行修改控制器")
    print("\n情况二：自行修改控制器")
    for line in structure2:
        print(f"  {line}")
    
    # 5. 获取必须文件
    print("\n4. 必须包含的文件：")
    required_files = requirements.get_required_files()
    for file in required_files:
        print(f"  - {file}")
    
    # 6. 获取命名规范
    print("\n5. 命名规范：")
    naming = requirements.get_naming_convention()
    print(f"  {naming}")
    
    # 7. 获取环境兼容性要求
    print("\n6. 环境兼容性要求：")
    compatibility = requirements.get_environment_compatibility()
    print(f"  {compatibility}")
    
    # 8. 获取严禁事项
    print("\n7. 严禁事项：")
    forbidden_actions = requirements.get_forbidden_actions()
    for action in forbidden_actions:
        print(f"  - {action}")
    
    # 9. 获取通用要求
    print("\n8. 通用要求：")
    general_reqs = requirements.get_general_requirements()
    for req in general_reqs:
        print(f"  - {req}")
    
    # 10. 获取Docker环境要求
    print("\n9. Docker环境要求：")
    docker_reqs = requirements.get_docker_requirements()
    for key, value in docker_reqs.items():
        print(f"  {key}: {value}")
    
    # 11. 测试提交验证功能
    print("\n10. 测试提交验证功能：")
    
    # 创建测试目录结构
    test_path = "/tmp/test_submission"
    os.makedirs(os.path.join(test_path, "craic_task_template", "scripts"), exist_ok=True)
    
    # 创建测试文件
    with open(os.path.join(test_path, "craic_task_template", "scripts", "scene1_patrol.py"), "w") as f:
        f.write("# Test scene1_patrol.py")
    
    with open(os.path.join(test_path, "craic_task_template", "scripts", "scene2_sorting.py"), "w") as f:
        f.write("# Test scene2_sorting.py")
    
    # 验证提交
    results = requirements.validate_submission(test_path)
    print(f"验证结果:")
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_path)
    
    print("\n=== 示例结束 ===")


if __name__ == "__main__":
    main()