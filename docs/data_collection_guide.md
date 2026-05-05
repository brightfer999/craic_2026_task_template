# 数据采集操作流程

## 一、实时显示指标说明

启动 `keyboard_teleop.py` 后，终端会显示以下实时指标：

```
======================================================================
  ROS 遥控  [SLOW]  vx=+0.15  vy=+0.00  wz=+0.00  yaw=+045°
----------------------------------------------------------------------
  CLEARANCE  前  :1.24m  左前:1.10m  右前:1.30m  左  :0.82m  右  :1.55m  后  :2.80m
  CONES      1.2m@+30°  |  2.5m@-45°
  STATE      wz≠0 ○  |  vy≠0 ○  |  运动:直行
----------------------------------------------------------------------
  w/x:前后  a/d:转向  q/e:横移  s:停  Space:变速  Ctrl+C:退出
======================================================================
```

### 各指标含义

| 指标 | 含义 | 说明 |
|------|------|------|
| **vx** | 前进速度 (m/s) | +前进 / -后退 |
| **vy** | 侧移速度 (m/s) | +左移 / -右移 |
| **wz** | 转向角速度 (rad/s) | +左转 / -右转 |
| **yaw** | 机器人朝向角 (°) | 0°=正东, ±180° |
| **前** | 前方 clearance (±18°) | --- 表示 >4.5m 无障碍 |
| **左前** | 左前 clearance (18°~68°) | 最近障碍物距离 |
| **右前** | 右前 clearance (-68°~-18°) | 最近障碍物距离 |
| **左** | 左侧 clearance (68°~120°) | 最近障碍物距离 |
| **右** | 右侧 clearance (-120°~-68°) | 最近障碍物距离 |
| **后** | 后方 clearance (150°~180°) | 最近障碍物距离 |
| **CONES** | 检测到的锥桶 | `距离@角度`，最多显示3个 |
| **wz≠0 ●/○** | 是否在转弯 | ●=正在转弯, ○=未转弯 |
| **vy≠0 ●/○** | 是否在侧移 | ●=正在侧移, ○=未侧移 |
| **转弯前距** | 转弯时前方 clearance | 仅在 wz≠0 时显示 |
| **⚠靠近锥桶** | 距最近锥桶 <1.0m | 自动告警 |
| **运动** | 运动分类 | 直行/原地转/前+转/侧移/静止 |

---

## 二、操作环境准备

需要开 **3 个终端窗口**：

```
┌─────────────────────┐  ┌─────────────────────┐  ┌─────────────────────┐
│  终端 1: 仿真       │  │  终端 2: 键盘遥控   │  │  终端 3: 数据采集   │
│                     │  │                     │  │                     │
│ python3 scripts/    │  │ python3 scripts/    │  │ python3 scripts/    │
│   scene1_patrol.py  │  │   keyboard_teleop.py│  │   data_collector.py │
│   --seed 0          │  │                     │  │   --output data/xx  │
│                     │  │  [看实时指标]       │  │                     │
│  (保持运行)         │  │  (操纵机器人)       │  │  (Ctrl+C 保存)     │
└─────────────────────┘  └─────────────────────┘  └─────────────────────┘
```

### 启动顺序

```bash
# === 步骤 1：终端 1 — 启动仿真 ===
cd /root/kuavo_ws/src/craic_task_template
python3 scripts/scene1_patrol.py --seed 0
# 等待约 20s，看到 "仿真环境就绪，机器人保持静止" 即可

# === 步骤 2：终端 2 — 启动键盘遥控 ===
cd /root/kuavo_ws/src/craic_task_template
python3 scripts/keyboard_teleop.py
# 看到多行显示界面即可开始操控

# === 步骤 3：终端 3 — 启动数据采集 ===
cd /root/kuavo_ws/src/craic_task_template
python3 scripts/data_collector.py --rate 10 --output data/场景名.npz
# 看到 "数据采集器启动" 即开始记录
```

---

## 三、各场景采集流程

### 场景 1：直行 (straight)

> 目标：记录空旷直行时的 lidar 数据

```
操作：
  按住 w 键，机器人前进 2~3 米后按 s 停止
  终端 3 Ctrl+C 保存

指标特征：
  前: 4.5m (---)   左右: 1.0~2.0m   锥桶: 无   运动:直行

文件命名：
  --output data/straight_clear.npz
```

### 场景 2：前方遇障 (front_blocked)

> 目标：采集前方不同距离有障碍物时的数据

```
操作：
  1. 驱动机器人正面朝向障碍物（墙壁/锥桶）
  2. 慢慢靠近，在 0.8m → 0.5m → 0.3m 各停留 2 秒
  3. 终端 3 Ctrl+C 保存

指标特征：
  前: 0.3~0.8m (逐步减小)   运动:直行

文件命名：
  --output data/front_blocked_03_08m.npz
```

### 场景 3：左侧有障 (left_blocked)

> 目标：采集左侧有墙壁/锥桶时的数据

```
操作：
  1. 驱动机器人使障碍物在左侧 0.3~0.8m 处
  2. 沿障碍物边缘直行 2~3 米（q 键左移贴近）
  3. 终端 3 Ctrl+C 保存

指标特征：
  左: 0.3~0.8m (很近)   前: 1.5m+   锥桶: 可能有   运动:直行/侧移

文件命名：
  --output data/left_blocked_wall.npz
```

### 场景 4：右侧有障 (right_blocked)

> 目标：采集右侧有墙壁/锥桶时的数据

```
操作：
  1. 驱动机器人使障碍物在右侧 0.3~0.8m 处
  2. 沿障碍物边缘直行 2~3 米（e 键右移贴近）
  3. 终端 3 Ctrl+C 保存

指标特征：
  右: 0.3~0.8m (很近)   前: 1.5m+   运动:直行/侧移

文件命名：
  --output data/right_blocked_wall.npz
```

### 场景 5：转弯避障 (turning_evasion)

> 目标：采集转弯时（wz≠0）的 lidar 数据

```
操作：
  1. 驱动机器人正面朝向锥桶/墙壁，靠近到 0.5m
  2. 按 a 或 d 键原地转弯 90°~180°
  3. 转弯过程中变化视角看障碍物
  4. 终端 3 Ctrl+C 保存

指标特征：
  wz≠0 ●转弯中   转弯前距:0.5m   yaw 持续变化   运动:原地转/前+转

文件命名：
  --output data/turn_evasion_90deg.npz
```

### 场景 6：侧移贴近 (lateral_approach)

> 目标：采集侧移时（vy≠0）的 lidar 数据

```
操作：
  1. 驱动机器人与墙壁平行，侧面距离约 1.0m
  2. 按 q 键左移贴近墙壁，直到 0.3~0.5m 按 s 停止
  3. 终端 3 Ctrl+C 保存

指标特征：
  vy≠0 ●侧移中   左: 0.3→1.0m (逐渐减小)   运动:侧移

文件命名：
  --output data/lateral_approach.npz
```

### 场景 7：锥桶多方角度 (cone_multi_angle)

> 目标：从不同角度采集锥桶数据 (0° / ±30° / ±60°)

```
操作：
  1. 找到场地中的锥桶，正对锥桶 (yaw 对准，CONES 显示 0°)
     → 停留 2s，靠近到 0.5m 再后退
  2. 侧移/转向，使锥桶出现在左前方 ~30° (CONES 显示 +30°)
     → 停留 2s
  3. 继续转到 ~60° (CONES 显示 +60°)
     → 停留 2s
  4. 同样采集 -30°、-60°
  5. 终端 3 Ctrl+C 保存

指标特征：
  CONES: 0.5m@0° → 0.5m@+30° → 0.5m@+60° → 0.5m@-30° → ...
  锥桶角度在 CONES 行直接可见

文件命名：
  --output data/cone_multi_angle.npz
```

### 场景 8：走廊居中 (corridor_center)

> 目标：采集走廊中居中行走的数据

```
操作：
  1. 驱动机器人到走廊中间位置
  2. 保持左右 clearance 大致相等（左≈右），直行 3~5 米
  3. 终端 3 Ctrl+C 保存

指标特征：
  左: 0.8~1.5m   右: 0.8~1.5m   左≈右   前: 2m+   运动:直行

文件命名：
  --output data/corridor_center.npz
```

### 场景 9：两侧受阻 (both_blocked)

> 目标：采集两侧都有障碍物（窄通道）的数据

```
操作：
  1. 驱动机器人进入窄通道（左右 clearance 都 <0.8m）
  2. 缓慢前进 2 米
  3. 终端 3 Ctrl+C 保存

指标特征：
  左: 0.3~0.8m   右: 0.3~0.8m   前: 1~3m   运动:直行

文件命名：
  --output data/both_sides_blocked.npz
```

---

## 四、停止与文件管理

### 停止采集

```
在终端 3 按 Ctrl+C → 数据自动保存到 --output 指定的路径
```

终端会显示保存确认：
```
  数据已保存: /root/kuavo_ws/.../data/front_blocked.npz  (12.3 MB, 234 帧, 23.4 秒)
```

### 文件存放位置

```
/root/kuavo_ws/src/craic_task_template/data/
├── straight_clear.npz
├── front_blocked_03_08m.npz
├── left_blocked_wall.npz
├── right_blocked_wall.npz
├── turn_evasion_90deg.npz
├── lateral_approach.npz
├── cone_multi_angle.npz
├── corridor_center.npz
└── both_sides_blocked.npz
```

### 事后改名

```bash
cd /root/kuavo_ws/src/craic_task_template/data

# 如果采集时忘了指定 --output，默认文件名带时间戳：
ls scene1_frames_*.npz

# 改名为有意义的名字：
mv scene1_frames_20260505_143022.npz  front_blocked_05m.npz
```

---

## 五、采集清单 (Checklist)

| # | 场景 | 文件名 | 关键指标 | ✓ |
|---|------|--------|----------|---|
| 1 | 直行空旷 | straight_clear.npz | 前:--- 运动:直行 | ☐ |
| 2 | 前方遇障 | front_blocked_03_08m.npz | 前:0.3→0.8m | ☐ |
| 3 | 左侧贴墙 | left_blocked_wall.npz | 左:<0.8m | ☐ |
| 4 | 右侧贴墙 | right_blocked_wall.npz | 右:<0.8m | ☐ |
| 5 | 转弯避障 | turn_evasion_90deg.npz | wz≠0 ● 转弯前距:<1m | ☐ |
| 6 | 侧移贴近 | lateral_approach.npz | vy≠0 ● | ☐ |
| 7 | 锥桶多角度 | cone_multi_angle.npz | CONES 0°→±30°→±60° | ☐ |
| 8 | 走廊居中 | corridor_center.npz | 左≈右 前>2m | ☐ |
| 9 | 两侧受阻 | both_sides_blocked.npz | 左<0.8m 右<0.8m | ☐ |

---

## 六、常见问题

**Q: 终端显示乱码/重叠？**
A: 确保终端窗口至少有 24 行 × 80 列。如果窗口太小，ANSI 光标定位会出错。

**Q: 激光雷达没有数据？**
A: 检查终端 2 的等待日志。如果 "未收到激光雷达数据"，可能是仿真还没完全启动，等几秒重试。

**Q: 锥桶检测不灵敏？**
A: 锥桶检测参数可调整 (CONE_MIN_POINTS, CONE_MAX_RADIUS 等)。默认检测距离 <3.0m 内的锥桶。

**Q: Ctrl+C 后文件没保存？**
A: data_collector 的 SIGINT 处理可能被 ROS 捕获。多按一次 Ctrl+C，或检查终端输出。
