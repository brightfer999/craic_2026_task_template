#!/usr/bin/env python3.8
"""
ROS 键盘遥控工具 — 直接发布 /cmd_vel 控制机器人。
集成实时激光雷达反馈：六扇区 clearance、锥桶检测、角度显示。

控制按键:
  w/x    前进/后退
  a/d    左转/右转
  q/e    左移/右移
  s      停止
  Space  切换速度档位 (慢速 0.15 / 快速 0.35 m/s)
  Ctrl+C 退出

实时显示指标:
  - 六扇区 clearance: 前 / 左前 / 右前 / 左 / 右 / 后
  - 锥桶检测: 最近锥桶的距离 + 角度
  - 机器人 yaw 角度 (0°~±180°)
  - 运动状态: wz≠0 转弯中, vy≠0 侧移中

使用方式:
  python3 scripts/keyboard_teleop.py
"""

import math
import os
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points

# ============================================================
# 显示常量
# ============================================================

SLOW_SPEED = 0.15
FAST_SPEED = 0.35
ANGULAR_SPEED = 0.50

# 激光雷达扇区定义 (与 LidarMapper 一致)
SECTORS = {
    "front":       (-math.radians(18),  math.radians(18)),
    "left_front":  ( math.radians(18),  math.radians(68)),
    "right_front": (-math.radians(68), -math.radians(18)),
    "left":        ( math.radians(68),  math.radians(120)),
    "right":       (-math.radians(120), -math.radians(68)),
    "rear":        ( math.radians(150), math.pi),
}

MAX_LIDAR_RANGE = 4.5

# 锥桶检测参数
CONE_MAX_SPAN = 0.55         # 聚类最大跨度 (m)
CONE_MAX_RADIUS = 0.32       # 聚类最大半径 (m)
CONE_MIN_POINTS = 3          # 最少点数
CONE_CLUSTER_DIAMETER = 0.55  # BFS 邻近距离阈值 (m)
CONE_DETECT_RANGE = 3.0      # 锥桶检测范围 (m)

# 显示行数 (用于 ANSI 光标上移)
DISPLAY_LINES = 9


def get_key(timeout_val=0.05):
    """非阻塞读取单个按键（必须在 tty.setraw() 之后调用）。"""
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout_val)
    if rlist:
        key = os.read(fd, 1)
        return key.decode() if isinstance(key, bytes) else key
    return ""


class KeyboardTeleop:
    """集成实时感知反馈的键盘遥控器"""

    def __init__(self):
        rospy.init_node("keyboard_teleop", anonymous=True)
        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._speed = SLOW_SPEED
        self._running = True
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0

        # ---- 按键诊断 ----
        self._last_key = ""       # 最近一次有效按键
        self._key_count = 0       # 累计按键次数

        # ---- 激光雷达数据 ----
        self._latest_cloud = None
        self._cloud_sub = rospy.Subscriber(
            "/lidar/points", PointCloud2, self._cloud_cb, queue_size=5
        )

        # ---- 里程计数据 ----
        self._latest_odom = None
        self._odom_sub = rospy.Subscriber(
            "/odom", Odometry, self._odom_cb, queue_size=5
        )

        # 等待首批数据
        rospy.loginfo("等待激光雷达和里程计数据...")
        waited = 0.0
        while (self._latest_cloud is None or self._latest_odom is None) and \
              not rospy.is_shutdown() and waited < 10.0:
            rospy.sleep(0.5)
            waited += 0.5
        if self._latest_cloud is None:
            rospy.logwarn("未收到激光雷达数据，将不显示 clearance")
        if self._latest_odom is None:
            rospy.logwarn("未收到里程计数据，将不显示角度")

    # ---- 回调 ----
    def _cloud_cb(self, msg):
        self._latest_cloud = msg

    def _odom_cb(self, msg):
        self._latest_odom = msg

    # ---- 按键处理 ----
    def process_key(self, key):
        # 诊断：记录每次有效按键
        if key and key != "\x03":
            self._last_key = key
            self._key_count += 1
            if self._key_count == 1:
                rospy.loginfo("  检测到首次按键 '%s' — 键盘输入正常工作", key)

        if key == "w":
            self._vx = self._speed
            self._wz = 0.0
        elif key == "x":
            self._vx = -self._speed
            self._wz = 0.0
        elif key == "a":
            self._wz = ANGULAR_SPEED
            if self._vx == 0.0:
                self._vx = self._speed * 0.3
        elif key == "d":
            self._wz = -ANGULAR_SPEED
            if self._vx == 0.0:
                self._vx = self._speed * 0.3
        elif key == "q":
            self._vy = self._speed
            self._wz = 0.0
        elif key == "e":
            self._vy = -self._speed
            self._wz = 0.0
        elif key == "s":
            self._vx = 0.0
            self._vy = 0.0
            self._wz = 0.0
        elif key == " ":
            self._speed = FAST_SPEED if self._speed == SLOW_SPEED else SLOW_SPEED
            return  # 不覆盖 _vx/_wz，保持当前运动
        elif key == "\x03":  # Ctrl+C
            self._running = False
            return
        elif key == "":
            pass
        else:
            return

    def publish(self):
        cmd = Twist()
        cmd.linear.x = self._vx
        cmd.linear.y = self._vy
        cmd.angular.z = self._wz
        self._pub.publish(cmd)

    # ---- 点云处理 ----
    def _get_points(self, max_range=MAX_LIDAR_RANGE, max_points=3000):
        """提取过滤后的点云列表 [(x, y, z), ...]"""
        cloud = self._latest_cloud
        if cloud is None:
            return []
        points = []
        for i, p in enumerate(read_points(cloud, field_names=("x", "y", "z"),
                                          skip_nans=True)):
            if i >= max_points:
                break
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            d = math.hypot(x, y)
            if 0.12 <= d <= max_range and -0.25 <= z <= 1.25:
                points.append((x, y, z))
        return points

    def _compute_clearance(self, points):
        """计算六扇区 clearance。"""
        result = {name: MAX_LIDAR_RANGE for name in SECTORS}
        if not points:
            return result

        for x, y, _z in points:
            angle = math.atan2(y, x)
            d = math.hypot(x, y)

            for name, (low, high) in SECTORS.items():
                if name == "rear":
                    in_sector = (angle >= low) or (angle <= -low)
                else:
                    in_sector = (low <= angle <= high)
                if in_sector and d < result[name]:
                    result[name] = d
        return result

    def _detect_cones(self, points):
        """BFS 聚类检测锥桶。

        锥桶特征：小尺寸、密集的点云簇。
        返回: [(cx, cy, distance, angle_deg), ...] 按距离排序。
        """
        if len(points) < CONE_MIN_POINTS:
            return []

        # 只取检测范围内的 XY
        nearby = [(x, y) for x, y, _z in points
                  if math.hypot(x, y) <= CONE_DETECT_RANGE]
        if len(nearby) < CONE_MIN_POINTS:
            return []

        n = len(nearby)

        # ---- 空间网格加速邻接构建 ----
        cell_size = CONE_CLUSTER_DIAMETER
        grid = {}  # (cx_cell, cy_cell) -> [index, ...]
        for i, (x, y) in enumerate(nearby):
            cx = int(x / cell_size)
            cy = int(y / cell_size)
            grid.setdefault((cx, cy), []).append(i)

        # 构建邻接表
        adj = [[] for _ in range(n)]
        for i, (xi, yi) in enumerate(nearby):
            ci_x = int(xi / cell_size)
            ci_y = int(yi / cell_size)
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for j in grid.get((ci_x + dx, ci_y + dy), []):
                        if j <= i:
                            continue
                        xj, yj = nearby[j]
                        if math.hypot(xi - xj, yi - yj) < CONE_CLUSTER_DIAMETER:
                            adj[i].append(j)
                            adj[j].append(i)

        # ---- BFS 找连通分量 ----
        visited = [False] * n
        cones = []

        for i in range(n):
            if visited[i]:
                continue
            cluster_idx = []
            queue = [i]
            visited[i] = True
            while queue:
                u = queue.pop(0)
                cluster_idx.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        queue.append(v)

            if len(cluster_idx) < CONE_MIN_POINTS:
                continue

            xs = [nearby[idx][0] for idx in cluster_idx]
            ys = [nearby[idx][1] for idx in cluster_idx]
            cx_val = sum(xs) / len(xs)
            cy_val = sum(ys) / len(ys)

            max_span = 0.0
            for a_idx in range(len(cluster_idx)):
                for b_idx in range(a_idx + 1, len(cluster_idx)):
                    span = math.hypot(xs[a_idx] - xs[b_idx],
                                     ys[a_idx] - ys[b_idx])
                    if span > max_span:
                        max_span = span

            max_radius = max(
                math.hypot(xs[k] - cx_val, ys[k] - cy_val)
                for k in range(len(cluster_idx))
            )

            if max_span <= CONE_MAX_SPAN and max_radius <= CONE_MAX_RADIUS:
                dist = math.hypot(cx_val, cy_val)
                angle = math.degrees(math.atan2(cy_val, cx_val))
                cones.append((cx_val, cy_val, dist, angle))

        cones.sort(key=lambda c: c[2])
        return cones

    # ---- 显示 ----
    def _get_yaw_deg(self):
        """从里程计提取 yaw 角度 (度, ±180°)。"""
        odom = self._latest_odom
        if odom is None:
            return None
        q = odom.pose.pose.orientation
        siny = 2.0 * (q.w * q.z + q.x * q.y)
        cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
        return math.degrees(math.atan2(siny, cosy))

    def _build_display(self, clearance, cones):
        """构建多行显示字符串。"""
        mode = "FAST" if self._speed == FAST_SPEED else "SLOW"
        yaw = self._get_yaw_deg()

        yaw_str = f"yaw={yaw:+04.0f}°" if yaw is not None else "yaw=---°"

        lines = []
        lines.append("=" * 68)
        lines.append(
            f"  ROS 遥控  [{mode}]  "
            f"vx={self._vx:+.2f}  vy={self._vy:+.2f}  wz={self._wz:+.2f}  "
            f"{yaw_str}"
        )
        lines.append("-" * 68)

        # ---- clearance 行 ----
        def fmt_clr(name, val):
            if val >= MAX_LIDAR_RANGE - 0.05:
                return f"{name}: --- "
            return f"{name}:{val:.2f}m"

        clr_parts = [
            fmt_clr("前  ", clearance["front"]),
            fmt_clr("左前", clearance["left_front"]),
            fmt_clr("右前", clearance["right_front"]),
            fmt_clr("左  ", clearance["left"]),
            fmt_clr("右  ", clearance["right"]),
            fmt_clr("后  ", clearance["rear"]),
        ]
        lines.append("  CLEARANCE  " + " ".join(clr_parts))

        # ---- 锥桶行 ----
        if cones:
            cone_parts = []
            for _, _, dist, ang in cones[:3]:
                cone_parts.append(f"{dist:.1f}m@{ang:+.0f}°")
            lines.append("  CONES      " + "  |  ".join(cone_parts))
        else:
            lines.append("  CONES      无")

        # ---- 状态行 ----
        state_parts = []
        if abs(self._wz) > 0.001:
            state_parts.append("wz≠0 ●转弯中")
        else:
            state_parts.append("wz≠0 ○")
        if abs(self._vy) > 0.001:
            state_parts.append("vy≠0 ●侧移中")
        else:
            state_parts.append("vy≠0 ○")

        # 运动分类
        if abs(self._vx) > 0.001 and abs(self._wz) > 0.001:
            motion = "前+转"
        elif abs(self._wz) > 0.001:
            motion = "原地转"
        elif abs(self._vx) > 0.001:
            motion = "直行"
        elif abs(self._vy) > 0.001:
            motion = "侧移"
        else:
            motion = "静止"

        # 前方 clearance 特殊标注 (wz≠0 时)
        if abs(self._wz) > 0.001:
            fclr = clearance["front"]
            if fclr < MAX_LIDAR_RANGE - 0.05:
                state_parts.append(f"转弯前距:{fclr:.2f}m")

        # 如果靠近锥桶
        if cones and cones[0][2] < 1.0:
            state_parts.append(f"⚠靠近锥桶{cones[0][2]:.1f}m")

        state_parts.append(f"运动:{motion}")
        lines.append("  STATE      " + "  |  ".join(state_parts))

        lines.append("-" * 68)
        # 按键提示 + 诊断
        if self._last_key:
            key_disp = f"最后按键: [{self._last_key}] (共{self._key_count}次)"
        else:
            key_disp = "等待按键输入..."
        lines.append(f"  w/x:前后  a/d:转向  q/e:横移  s:停  Space:变速  Ctrl+C:退出  |  {key_disp}")
        lines.append("=" * 68)

        return "\r\n".join(lines)

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())

        # 预留显示空间
        for _ in range(DISPLAY_LINES):
            print("")

        rate = rospy.Rate(20)
        first_display = True

        try:
            while self._running and not rospy.is_shutdown():
                # 1. 读取按键
                key = get_key()
                if key:
                    self.process_key(key)

                # 2. 发布速度指令
                self.publish()

                # 3. 计算感知数据
                points = self._get_points()
                clearance = self._compute_clearance(points)
                cones = self._detect_cones(points)

                # 4. 刷新多行显示
                if first_display:
                    first_display = False
                else:
                    sys.stdout.write(f"\033[{DISPLAY_LINES}A")

                display = self._build_display(clearance, cones)
                sys.stdout.write(display)
                sys.stdout.flush()

                rate.sleep()

        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            cmd = Twist()
            self._pub.publish(cmd)
            print("\n\n  已停止，退出。")


if __name__ == "__main__":
    try:
        KeyboardTeleop().run()
    except rospy.ROSInterruptException:
        pass
