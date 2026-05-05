#!/usr/bin/env python3.8
"""
ROS 键盘遥控工具 — 直接发布 /cmd_vel 控制机器人。

控制按键:
  w/x    前进/后退
  a/d    左转/右转
  q/e    左移/右移
  s      停止
  Space  切换速度档位 (慢速 0.15 / 快速 0.35 m/s)
  Ctrl+C 退出

使用方式:
  python3 scripts/keyboard_teleop.py
"""

import os
import select
import sys
import termios
import tty

import rospy
from geometry_msgs.msg import Twist


BANNER = """
  ROS 键盘遥控 — /cmd_vel 发布中
  ─────────────────────────────────────
    w/x : 前进/后退    q/e : 左移/右移
    a/d : 左转/右转    s   : 停止
    Space : 切换速度档位
    Ctrl+C : 退出
  ─────────────────────────────────────
"""

SLOW_SPEED = 0.15
FAST_SPEED = 0.35
ANGULAR_SPEED = 0.50


def get_key(settings, timeout=0.05):
    """非阻塞读取单个按键。"""
    fd = sys.stdin.fileno()
    rlist, _, _ = select.select([fd], [], [], timeout)
    if rlist:
        key = os.read(fd, 1)
        termios.tcsetattr(fd, termios.TCSADRAIN, settings)
        return key.decode() if isinstance(key, bytes) else key
    return ""


class KeyboardTeleop:
    def __init__(self):
        rospy.init_node("keyboard_teleop", anonymous=True)
        self._pub = rospy.Publisher("/cmd_vel", Twist, queue_size=10)
        self._speed = SLOW_SPEED
        self._running = True
        self._vx = 0.0
        self._vy = 0.0
        self._wz = 0.0

    def process_key(self, key):
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
            rospy.loginfo("速度档位切换到: %.2f m/s", self._speed)
            return
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

    def status_line(self):
        mode = "FAST" if self._speed == FAST_SPEED else "SLOW"
        return f"\r  [{mode}] vx={self._vx:+.2f} vy={self._vy:+.2f} wz={self._wz:+.2f}  "

    def run(self):
        settings = termios.tcgetattr(sys.stdin)
        tty.setraw(sys.stdin.fileno())
        print(BANNER)

        rate = rospy.Rate(20)
        try:
            while self._running and not rospy.is_shutdown():
                key = get_key(settings)
                if key:
                    self.process_key(key)
                self.publish()
                sys.stdout.write(self.status_line())
                sys.stdout.flush()
                rate.sleep()
        finally:
            termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
            cmd = Twist()
            self._pub.publish(cmd)
            print("\n  已停止，退出。")


if __name__ == "__main__":
    try:
        KeyboardTeleop().run()
    except rospy.ROSInterruptException:
        pass
