"""
CRAIC 2026 仿真赛比赛要求封装模块
提供比赛规则、约束条件、文件结构要求等信息
"""

import os
from typing import Dict, List, Optional
from dataclasses import dataclass


@dataclass
class CompetitionRules:
    """比赛规则数据类"""
    allowed_sensors: List[str]
    forbidden_interfaces: List[str]
    allowed_modification_scope: List[str]
    navigation_principles: List[str]


@dataclass
class SubmissionRequirements:
    """提交要求数据类"""
    file_structure: Dict[str, List[str]]
    naming_convention: str
    required_files: List[str]
    optional_packages: List[str]
    environment_compatibility: str


class CompetitionRequirements:
    """
    CRAIC 2026 仿真赛比赛要求封装类
    提供比赛规则、约束条件、文件结构要求等信息
    """
    
    def __init__(self):
        """初始化比赛要求"""
        self.rules = self._init_rules()
        self.submission_requirements = self._init_submission_requirements()
    
    def _init_rules(self) -> CompetitionRules:
        """初始化比赛规则"""
        return CompetitionRules(
            allowed_sensors=[
                "/odom",
                "/lidar/points", 
                "/tag_detections",
                "头部相机按钮识别"
            ],
            forbidden_interfaces=[
                "/ground_truth/state",
                "任何运行时真值状态"
            ],
            allowed_modification_scope=[
                "仅 src/craic_task_template 内的 README、脚本和辅助模块"
            ],
            navigation_principles=[
                "阶段顺序是已知任务指令",
                "具体移动由 AprilTag 锚点、局部雷达地图和相对行程共同决定"
            ]
        )
    
    def _init_submission_requirements(self) -> SubmissionRequirements:
        """初始化提交要求"""
        return SubmissionRequirements(
            file_structure={
                "情况一：使用仿真环境自带控制器": [
                    "参赛团队名称+人形机器人创新挑战赛/",
                    "├── craic_task_template/            # 功能包名称固定",
                    "│   ├── CMakeLists.txt",
                    "│   ├── package.xml",
                    "│   ├── scripts/                    # 任务脚本目录（必须）",
                    "│   │   ├── scene1_patrol.py        # 场景一：安全巡检任务脚本",
                    "│   │   └── scene2_sorting.py       # 场景二：分拣归档任务脚本",
                    "│   ├── launch/                     # launch 文件（可选）",
                    "│   ├── src/                        # 其他源码（可选）",
                    "│   └── README.md                   # 运行说明"
                ],
                "情况二：自行修改控制器": [
                    "参赛团队名称+人形机器人创新挑战赛/",
                    "├── craic_task_template/            # 任务功能包（必须）",
                    "│   ├── CMakeLists.txt",
                    "│   ├── package.xml",
                    "│   ├── scripts/",
                    "│   │   ├── scene1_patrol.py",
                    "│   │   └── scene2_sorting.py",
                    "│   └── README.md                   # 需说明修改了哪些功能包及原因",
                    "└── humanoid_controllers/           # 示例：修改过的控制器功能包",
                    "    └── ...                         # 仅包含修改过的功能包，保持原包名"
                ]
            },
            naming_convention="参赛团队名称+人形机器人创新挑战赛",
            required_files=[
                "scripts/scene1_patrol.py",
                "scripts/scene2_sorting.py"
            ],
            optional_packages=[
                "humanoid_controllers"
            ],
            environment_compatibility="Ubuntu 20.04 Docker容器"
        )
    
    def get_allowed_sensors(self) -> List[str]:
        """获取允许使用的传感器列表"""
        return self.rules.allowed_sensors
    
    def get_forbidden_interfaces(self) -> List[str]:
        """获取禁止使用的接口列表"""
        return self.rules.forbidden_interfaces
    
    def get_allowed_modification_scope(self) -> List[str]:
        """获取允许修改范围"""
        return self.rules.allowed_modification_scope
    
    def get_navigation_principles(self) -> List[str]:
        """获取导航原则"""
        return self.rules.navigation_principles
    
    def get_file_structure(self, situation: str = "情况一：使用仿真环境自带控制器") -> List[str]:
        """获取文件结构要求"""
        return self.submission_requirements.file_structure.get(situation, [])
    
    def get_naming_convention(self) -> str:
        """获取命名规范"""
        return self.submission_requirements.naming_convention
    
    def get_required_files(self) -> List[str]:
        """获取必须包含的文件列表"""
        return self.submission_requirements.required_files
    
    def get_environment_compatibility(self) -> str:
        """获取环境兼容性要求"""
        return self.submission_requirements.environment_compatibility
    
    def validate_submission(self, submission_path: str) -> Dict[str, bool]:
        """
        验证提交是否符合要求
        
        Args:
            submission_path: 提交文件夹路径
            
        Returns:
            验证结果字典
        """
        validation_results = {
            "has_craic_task_template": False,
            "has_scene1_patrol": False,
            "has_scene2_sorting": False,
            "naming_convention_correct": False,
            "environment_compatible": True  # 假设环境兼容
        }
        
        # 检查是否存在craic_task_template文件夹
        craic_template_path = os.path.join(submission_path, "craic_task_template")
        if os.path.exists(craic_template_path):
            validation_results["has_craic_task_template"] = True
            
            # 检查必须文件
            scene1_path = os.path.join(craic_template_path, "scripts", "scene1_patrol.py")
            scene2_path = os.path.join(craic_template_path, "scripts", "scene2_sorting.py")
            
            validation_results["has_scene1_patrol"] = os.path.exists(scene1_path)
            validation_results["has_scene2_sorting"] = os.path.exists(scene2_path)
        
        # 检查命名规范
        folder_name = os.path.basename(submission_path)
        validation_results["naming_convention_correct"] = folder_name.endswith("人形机器人创新挑战赛")
        
        return validation_results
    
    def get_forbidden_actions(self) -> List[str]:
        """获取严禁事项列表"""
        return [
            "严禁直接修改物品位置 — 物品必须通过机器人物理操作（抓取、搬运、放置）来移动，不得通过任何方式跳过操作过程",
            "严禁对机器人施加非物理外力 — 机器人运动必须依靠自身关节驱动，不得通过额外手段辅助",
            "严禁操控仿真运行状态 — 不得暂停、加速或以其他方式干预仿真进程",
            "严禁读取真值信息 — 机器人应通过自身传感器（雷达、IMU 等）感知环境，不得直接获取物体的绝对坐标",
            "严禁修改仿真环境文件 — 不得修改 craic_simulator 包中的场景文件、launch 文件或其他配置"
        ]
    
    def get_general_requirements(self) -> List[str]:
        """获取通用要求列表"""
        return [
            "功能包将被放置到仿真环境的 src/ 目录下，与 craic_simulator 同级，通过编译运行",
            "必须包含 scripts/scene1_patrol.py 和 scripts/scene2_sorting.py，缺失则对应任务失败",
            "正式比赛将运行这两个程序完成评分",
            "如需额外依赖，请在README.md 中注明"
        ]
    
    def get_docker_requirements(self) -> Dict[str, str]:
        """获取Docker环境要求"""
        return {
            "操作系统": "Ubuntu 20.04",
            "容器环境": "官方提供的 Docker 容器",
            "兼容性要求": "代码在官方提供的 Docker 容器中运行，请确保代码兼容该环境"
        }
    
    def print_summary(self):
        """打印比赛要求摘要"""
        print("=== CRAIC 2026 仿真赛比赛要求摘要 ===\n")
        
        print("1. 允许使用的传感器:")
        for sensor in self.rules.allowed_sensors:
            print(f"   - {sensor}")
        
        print("\n2. 禁止使用的接口:")
        for interface in self.rules.forbidden_interfaces:
            print(f"   - {interface}")
        
        print("\n3. 允许修改范围:")
        for scope in self.rules.allowed_modification_scope:
            print(f"   - {scope}")
        
        print("\n4. 导航原则:")
        for principle in self.rules.navigation_principles:
            print(f"   - {principle}")
        
        print("\n5. 必须包含的文件:")
        for file in self.submission_requirements.required_files:
            print(f"   - {file}")
        
        print("\n6. 命名规范:")
        print(f"   - {self.submission_requirements.naming_convention}")
        
        print("\n7. 环境兼容性:")
        print(f"   - {self.submission_requirements.environment_compatibility}")
        
        print("\n8. 严禁事项:")
        for action in self.get_forbidden_actions():
            print(f"   - {action}")
        
        print("\n9. 通用要求:")
        for req in self.get_general_requirements():
            print(f"   - {req}")


# 创建全局实例
competition_requirements = CompetitionRequirements()


def get_competition_requirements() -> CompetitionRequirements:
    """获取比赛要求实例"""
    return competition_requirements


if __name__ == "__main__":
    # 测试比赛要求模块
    requirements = get_competition_requirements()
    requirements.print_summary()
    
    # 测试验证功能
    print("\n=== 测试验证功能 ===")
    test_path = "/tmp/test_submission"
    os.makedirs(os.path.join(test_path, "craic_task_template", "scripts"), exist_ok=True)
    
    # 创建测试文件
    with open(os.path.join(test_path, "craic_task_template", "scripts", "scene1_patrol.py"), "w") as f:
        f.write("# Test scene1_patrol.py")
    
    with open(os.path.join(test_path, "craic_task_template", "scripts", "scene2_sorting.py"), "w") as f:
        f.write("# Test scene2_sorting.py")
    
    # 验证提交
    results = requirements.validate_submission(test_path)
    print(f"验证结果: {results}")
    
    # 清理测试文件
    import shutil
    shutil.rmtree(test_path)