#!/usr/bin/env python3
"""
提交验证脚本
使用CompetitionRequirements模块验证提交是否符合比赛要求
"""

import sys
import os
import argparse

# 添加skills目录到Python路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from skills import CompetitionRequirements


def validate_submission(submission_path: str) -> bool:
    """
    验证提交是否符合要求
    
    Args:
        submission_path: 提交文件夹路径
        
    Returns:
        验证是否通过
    """
    print(f"验证提交: {submission_path}\n")
    
    # 创建比赛要求实例
    requirements = CompetitionRequirements()
    
    # 验证提交
    results = requirements.validate_submission(submission_path)
    
    # 显示验证结果
    print("验证结果:")
    all_passed = True
    for key, value in results.items():
        status = "✓" if value else "✗"
        print(f"  {status} {key}: {value}")
        if not value:
            all_passed = False
    
    # 显示详细要求
    print("\n详细要求:")
    
    # 显示必须文件
    required_files = requirements.get_required_files()
    print(f"必须文件: {required_files}")
    
    # 显示命名规范
    naming = requirements.get_naming_convention()
    print(f"命名规范: {naming}")
    
    # 显示文件结构
    print("\n文件结构要求:")
    structure = requirements.get_file_structure()
    for line in structure:
        print(f"  {line}")
    
    # 显示严禁事项
    print("\n严禁事项:")
    forbidden_actions = requirements.get_forbidden_actions()
    for action in forbidden_actions:
        print(f"  - {action}")
    
    return all_passed


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="验证提交是否符合CRAIC 2026比赛要求")
    parser.add_argument("submission_path", help="提交文件夹路径")
    args = parser.parse_args()
    
    # 验证提交
    if validate_submission(args.submission_path):
        print("\n✓ 提交验证通过！")
        return 0
    else:
        print("\n✗ 提交验证失败！")
        return 1


if __name__ == "__main__":
    sys.exit(main())