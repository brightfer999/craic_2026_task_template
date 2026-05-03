# 2026 CRAIC 仿真赛环境与任务模板

本仓库包含 **2026 年 CRAIC 人形机器人仿真挑战赛** 的完整仿真环境及任务模板。

## 环境搭建

### 1. 导入 Docker 镜像
```bash
# 下载镜像
wget "https://kuavo.lejurobot.com/docker_images/2026%20craic%20%E4%BB%BF%E7%9C%9F%E6%AF%94%E8%B5%9B%E9%95%9C%E5%83%8F/kuavo_craic_img_v1.0.tar.gz"
# 导入镜像
docker load -i kuavo_craic_img_v1.0.tar.gz
```

### 2. 启动 Docker 容器
```bash
./docker/run_with_gpu_for_craic.sh
```
> 容器内自动设置 `ROBOT_VERSION=52`，工作目录映射到 `/root/kuavo_ws`。

### 3. 编译（容器内执行）
```bash
cd /root/kuavo_ws
catkin config -DCMAKE_ASM_COMPILER=/usr/bin/as -DCMAKE_BUILD_TYPE=Release
source installed/setup.zsh
catkin build craic_simulator kuavo_msgs humanoid_controllers
source devel/setup.zsh
```

## 比赛场景

### 场景一：安全巡检
路线：起点 → 障碍区 → 操作台1 → 复杂地形区 → 操作台2 → 斜坡区 → 台阶区 → 终点
附加任务（可选）：终点 → 台阶区 → 斜坡区 → 复杂地形区 → 障碍区 → 起点
```bash
python3 src/craic_task_template/scripts/scene1_patrol.py --seed 0
```

### 场景二：分拣归档
- 子任务一 - 料盘下架（40 分）：从料盘架夹取料盘放入物料箱
- 子任务二 - 分拣零件（30 分）：工作台上两种零件分类放入两侧物料箱
```bash
python3 src/craic_task_template/scripts/scene2_sorting.py --seed 0
```

### 随机种子
通过 `--seed` 参数控制随机化（机器人初始位姿、物体位置、按钮亮暗等），不同种子对应不同的场景配置。

### 场景二维码
场景中的楼梯、斜坡、料盘架等位置放置了 **AprilTag** 二维码标记，标签族为 **tag36h11**，ID 范围 1~8。

### 仿真计时器
比赛计时基于 MuJoCo 仿真时间（`sensor_time`），与电脑性能无关，确保所有选手公平。在另一个终端运行：
```bash
python3 src/craic_simulator/scripts/sim_timer.py
```
会实时显示仿真时间和已用时间（秒）。

## 常用接口

> 完整接口文档请参考：[接口使用文档](https://kuavo.lejurobot.com/manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/)
>
> 除"严禁事项"中列出的违规行为外，选手可自由使用仿真环境中的其他接口，可通过 `rostopic list` 和 `rosservice list` 查看所有可用接口。

### 控制类
| 接口 | 类型 | 说明 |
|------|------|------|
| `cmd_vel` | geometry_msgs/Twist | 速度指令：`linear.x`=前进, `linear.y`=侧移, `angular.z`=转向 |
| `/kuavo_arm_traj` | sensor_msgs/JointState | 手臂轨迹控制（左臂7关节 + 右臂7关节） |
| `/gripper/command` | sensor_msgs/JointState | 夹爪命令（建议使用 GripperController） |
| `/humanoid_controller/switch_controller` | kuavo_msgs/switchController | 切换控制器（如 `amp_controller`） |

### 传感器类
| 接口 | 类型 | 说明 |
|------|------|------|
| `/sensors_data_raw` | kuavo_msgs/sensorsData | 传感器原始数据（IMU、关节位置/速度/力矩等） |
| `/lidar/points` | sensor_msgs/PointCloud2 | Mid360 雷达点云数据 |
| `/cam_h/color/image_raw/compressed` | sensor_msgs/CompressedImage | 头部 RGB 摄像头 |
| `/cam_l/color/image_raw/compressed` | sensor_msgs/CompressedImage | 左手腕 D405 RGB 摄像头 |
| `/cam_r/color/image_raw/compressed` | sensor_msgs/CompressedImage | 右手腕 D405 RGB 摄像头 |
| `/cam_h/color/camera_info` | sensor_msgs/CameraInfo | 头部摄像头内参 |
| `/cam_l/color/camera_info` | sensor_msgs/CameraInfo | 左手腕摄像头内参 |
| `/cam_r/color/camera_info` | sensor_msgs/CameraInfo | 右手腕摄像头内参 |
| `/cam_h/depth/image_raw/compressedDepth` | sensor_msgs/CompressedImage | 头部深度摄像头 |
| `/cam_l/depth/image_rect_raw/compressedDepth` | sensor_msgs/CompressedImage | 左手腕深度摄像头 |
| `/cam_r/depth/image_rect_raw/compressedDepth` | sensor_msgs/CompressedImage | 右手腕深度摄像头 |
| `/gripper/state` | sensor_msgs/JointState | 夹爪当前状态 |

> 注意：仿真环境中的摄像头和雷达 topic 名称与实物接口文档中有所不同，请以上表为准。其它接口文档参考：[接口使用文档](https://kuavo.lejurobot.com/manual/basic_usage/kuavo-ros-control/docs/4%E5%BC%80%E5%8F%91%E6%8E%A5%E5%8F%A3/%E6%8E%A5%E5%8F%A3%E4%BD%BF%E7%94%A8%E6%96%87%E6%A1%A3/)

### 夹爪控制器
```python
from gripper_controller import GripperController
gripper = GripperController()
gripper.open_grippers()  # 张开
gripper.close_grippers()  # 闭合
gripper.set_gripper_position(left=200, right=200)  # 指定开合度 (0-255)
```

## 评分细则与提交规范

详见 [评分细则与代码提交规范](src/craic_task_template/README.md)，包含：
- 场景一（50分）+ 场景二（70分）= 总分 120 分
- 代码提交格式、命名规范
- 严禁事项（违规取消成绩）

### 提交方式
通过报名系统提交附件，命名格式：**`参赛团队名称+人形机器人创新挑战赛`**
基于 `craic_task_template` 功能包开发，必须包含 `scripts/scene1_patrol.py` 和 `scripts/scene2_sorting.py`。

## 相关文档
- [评分细则与代码提交规范](src/craic_task_template/README.md)
- [选手脚本模板 - 场景一](src/craic_task_template/scripts/scene1_patrol.py)
- [选手脚本模板 - 场景二](src/craic_task_template/scripts/scene2_sorting.py)

---

## 场景一自动巡检实现说明

### 方案定位

当前实现采用"AprilTag 语义锚点 + 激光雷达局部实时建图 + 里程计短程相对位移"的混合感知导航方案。机器人不再依赖绝对世界坐标路点建模，也不读取 `/ground_truth/state`、不修改 `craic_simulator` 场景文件、不直接操作物体位置。

AprilTag 用来确认当前靠近的任务区域和阶段，激光雷达 `/lidar/points` 用来实时判断前方、左右前方和侧向通道是否可走，`/odom` 只用于阶段内的短距离行进估计。启动时即使还没有看到 AprilTag，也会根据雷达局部通行方向向障碍区试探前进，不会因为全局定位失败而停在起点。

### 全局指令

- 允许修改范围：仅 `src/craic_task_template` 内的 README、脚本和辅助模块。
- 允许使用传感器：`/odom`、`/lidar/points`、`/tag_detections`、头部相机按钮识别。
- 禁止使用接口：`/ground_truth/state` 和任何运行时真值状态。
- 导航原则：阶段顺序是已知任务指令，具体移动由 AprilTag 锚点、局部雷达地图和相对行程共同决定。

### 模块划分

- `scripts/scene1_patrol.py`：启动仿真，创建控制器、定位器和巡检状态机。
- `scripts/waypoints.py`：保存任务阶段、AprilTag 语义、相对行程阈值、控制器策略和按钮配置；不再作为绝对路径坐标表。
- `scripts/localization.py`：维护相对里程计、最近 AprilTag 锚点和阶段内行程，不估计全局 `world` 坐标。
- `scripts/local_lidar_map.py`：订阅 `/lidar/points`，生成局部扇区/栅格通行代价。
- `scripts/terrain_traverser.py`：根据局部雷达地图执行试探前进、障碍区穿越、走廊跟随、低速复杂地形通过和按钮操作。
- `scripts/state_machine.py`：按任务阶段串联动作，阶段切换依赖 Tag、相对行程和局部穿越结果。

### 运行流程

1. `SimLauncher(scene="scene1", seed=args.seed)` 启动仿真并初始化 ROS 节点。
2. `ControllerManager.use_mpc()` 切到平地控制器。
3. `Localizer.init_localize()` 等待 `/odom` 和可选 `/tag_detections`，没有 Tag 也允许启动。
4. `PatrolFSM.run()` 按"起点 → 障碍区 → 操作台1 → 复杂地形 → 操作台2 → 斜坡 → 台阶 → 终点"执行。
5. 若主任务稳定完成、时间充足且相对定位仍可信，则尝试反向附加任务。

### 控制器策略

- 起步、障碍区、操作台靠近、终点进入：使用 `mpc`，配合雷达局部避障。
- 复杂地形、斜坡、台阶：使用 `amp_controller` 或可用的低速步态控制器，以相对行程和雷达通道保持为主。
- AprilTag 可见时优先作为阶段锚点；不可见时按相对行程阈值降级推进。

### ROS 接口

- 发布 `/cmd_vel`：底盘/步态速度命令。
- 发布 `/kuavo_arm_traj`：通过 `ButtonPresser` 控制手臂按键。
- 订阅 `/odom`：阶段内相对行程和朝向变化。
- 订阅 `/lidar/points`：局部扇区/栅格地图和实时避障。
- 可选订阅 `/tag_detections`：AprilTag 语义锚点检测。
- 调用 `/humanoid_controller/switch_controller`：在 `mpc` 和 `amp_controller`/备选步态之间切换。
- 使用 `GripperController`：按钮按压时控制夹爪。

### 调试建议

1. 固定 seed 运行 `scene1_patrol.py --seed 0`，确认启动后不会因没有全局定位退出。
2. 查看 `Localizer` 日志，确认 `/odom` 可用、AprilTag 可见时能更新当前锚点。
3. 查看 `LocalLidarMapper` 日志，确认前方被挡时会选择左前或右前绕行。
4. 分段调试障碍区、操作台、复杂地形、斜坡和台阶的相对行程阈值。
5. 多 seed 验证启动试探前进和障碍区绕行稳定性。

---

## Skills 模块

`skills/` 目录包含可重用的技能模块，提供场景1所需的核心功能。

### 基础工具模块

#### `ROSInterface` - ROS接口封装
- 统一的ROS话题发布和订阅管理
- 支持速度控制、手臂控制、夹爪控制
- 传感器数据回调管理

#### `NavigationUtils` - 导航工具
- 距离和角度计算
- 路径插值和平滑
- 点与多边形关系判断

#### `PerceptionUtils` - 感知工具
- 点云数据处理（过滤、下采样）
- 障碍物提取和聚类
- 图像格式转换

### 场景1专用模块

#### `LidarMapper` - 激光雷达局部建图
```python
from skills import LidarMapper

lidar = LidarMapper()
clearance = lidar.clearance()  # 获取各扇区clearance
if clearance.front < 0.55:
    # 前方有障碍物
    turn = lidar.preferred_turn()  # 获取推荐转向
```

**功能**：
- 实时计算机器人周围6个扇区的clearance
- 提供走廊跟随误差计算
- 推荐避障转向方向
- 检测前方障碍物密度

#### `ButtonDetector` - 按钮检测
```python
from skills import ButtonDetector

detector = ButtonDetector()
if detector.is_lit("green"):
    print("绿色按钮亮起")
button = detector.detect_button("red")
if button:
    cx, cy, radius = button
    print(f"红色按钮位置: ({cx}, {cy})")
```

**功能**：
- HSV颜色空间按钮检测
- 支持红、绿、蓝、黄四种颜色
- 按钮亮起状态判断
- 按钮位置和大小检测

#### `Localizer` - 定位器
```python
from skills import Localizer

localizer = Localizer(apriltag_anchors={...})
localizer.init_localize()
pose = localizer.get_pose()  # 获取相对位姿
if localizer.has_seen_tag((1, 2, 3)):
    anchor = localizer.latest_anchor()
```

**功能**：
- 相对里程计跟踪（基于odom）
- AprilTag语义锚点检测
- 阶段内行程跟踪
- 位姿有效性验证

#### `ControllerManager` - 控制器管理
```python
from skills import ControllerManager

controller = ControllerManager()
controller.use_mpc()  # 切换到MPC控制器
controller.use_amp()  # 切换到AMP控制器
controller.send_velocity_command(linear_x=0.2)
controller.stop_robot()
```

**功能**：
- MPC/AMP控制器切换
- 自动降级机制（AMP不可用时使用leju_walk）
- 速度命令发布
- 机器人停止控制

### 使用示例

```python
from skills import LidarMapper, Localizer, ControllerManager, ButtonDetector

# 初始化模块
lidar = LidarMapper()
localizer = Localizer(apriltag_anchors={...})
controller = ControllerManager()
button_detector = ButtonDetector()

# 初始化定位
localizer.init_localize()

# 切换到MPC控制器
controller.use_mpc()

# 主循环
rate = rospy.Rate(20)
while not rospy.is_shutdown():
    # 获取传感器数据
    clearance = lidar.clearance()
    pose = localizer.get_pose()
    
    # 避障逻辑
    if clearance.front < 0.55:
        turn = lidar.preferred_turn()
        controller.send_velocity_command(angular_z=turn * 0.3)
    else:
        controller.send_velocity_command(linear_x=0.2)
    
    # 检查按钮
    if button_detector.is_lit("green"):
        rospy.loginfo("检测到绿色按钮亮起")
    
    rate.sleep()
```

### 模块依赖

- **ROS消息**：`geometry_msgs`, `sensor_msgs`, `nav_msgs`
- **第三方库**：`numpy`, `opencv-python`
- **可选依赖**：`apriltag_ros`（用于AprilTag检测）

## 比赛要求模块

### `CompetitionRequirements` - 比赛要求封装
```python
from skills import CompetitionRequirements

# 获取比赛要求实例
requirements = CompetitionRequirements()

# 打印比赛要求摘要
requirements.print_summary()

# 获取允许使用的传感器
sensors = requirements.get_allowed_sensors()

# 获取禁止使用的接口
forbidden = requirements.get_forbidden_interfaces()

# 验证提交是否符合要求
results = requirements.validate_submission("/path/to/submission")
```

**功能**：
- 比赛规则封装（允许传感器、禁止接口、修改范围）
- 提交要求管理（文件结构、命名规范、必须文件）
- 提交验证功能
- 严禁事项和通用要求
- Docker环境兼容性要求

**主要方法**：
- `get_allowed_sensors()`: 获取允许使用的传感器列表
- `get_forbidden_interfaces()`: 获取禁止使用的接口列表
- `get_allowed_modification_scope()`: 获取允许修改范围
- `get_navigation_principles()`: 获取导航原则
- `get_file_structure()`: 获取文件结构要求
- `validate_submission()`: 验证提交是否符合要求
- `get_forbidden_actions()`: 获取严禁事项列表
- `get_general_requirements()`: 获取通用要求列表
- `get_docker_requirements()`: 获取Docker环境要求