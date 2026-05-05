#!/usr/bin/env python3.8
"""
数据采集工具 — 同时记录激光雷达、速度指令、里程计数据，保存为 .npz 文件。

订阅话题:
  /lidar/points  (PointCloud2)  激光雷达点云
  /cmd_vel       (Twist)        速度控制指令
  /odom          (Odometry)     里程计位姿

使用方式:
  python3 scripts/data_collector.py                          # 默认保存路径
  python3 scripts/data_collector.py --output data/my.npz     # 指定输出文件
  python3 scripts/data_collector.py --rate 10                # 采集频率
  python3 scripts/data_collector.py --max-frames 200         # 最大帧数限制

按 Ctrl+C 停止采集，数据自动保存为 .npz 文件。
"""

import argparse
import os
import signal
import sys
import time

import numpy as np
import rospy
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2
from sensor_msgs.point_cloud2 import read_points


class DataCollector:
    def __init__(self, output_path, rate=10, max_frames=0):
        self._output_path = output_path
        self._rate = rate
        self._max_frames = max_frames
        self._frames = []
        self._start_time = None
        self._running = True

        self._latest_cloud = None
        self._latest_cmd = None
        self._latest_odom = None
        self._frame_count = 0

        rospy.init_node("data_collector", anonymous=True)
        self._cloud_sub = rospy.Subscriber(
            "/lidar/points", PointCloud2, self._cloud_cb, queue_size=5
        )
        self._cmd_sub = rospy.Subscriber(
            "/cmd_vel", Twist, self._cmd_cb, queue_size=5
        )
        self._odom_sub = rospy.Subscriber(
            "/odom", Odometry, self._odom_cb, queue_size=5
        )

        signal.signal(signal.SIGINT, self._signal_handler)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    def _cloud_cb(self, msg):
        self._latest_cloud = msg

    def _cmd_cb(self, msg):
        self._latest_cmd = msg

    def _odom_cb(self, msg):
        self._latest_odom = msg

    def _signal_handler(self, sig, frame):
        rospy.loginfo("\n  收到中断信号，正在保存数据...")
        self._running = False

    def _extract_lidar(self, cloud_msg):
        """提取点云 (x, y, z)，过滤 NaN 和过远点。"""
        pts = []
        for p in read_points(cloud_msg, field_names=("x", "y", "z"), skip_nans=True):
            x, y, z = float(p[0]), float(p[1]), float(p[2])
            d = np.hypot(x, y)
            if 0.10 <= d <= 4.5:
                pts.append([x, y, z])
        return np.array(pts, dtype=np.float32)

    def _capture_frame(self):
        cloud = self._latest_cloud
        cmd = self._latest_cmd
        odom = self._latest_odom

        if cloud is None:
            return

        lidar_pts = self._extract_lidar(cloud)

        cmd_vel = np.array(
            [cmd.linear.x, cmd.linear.y, cmd.linear.z,
             cmd.angular.x, cmd.angular.y, cmd.angular.z] if cmd else [0.0] * 6,
            dtype=np.float32,
        )

        if odom:
            odom_arr = np.array(
                [odom.pose.pose.position.x,
                 odom.pose.pose.position.y,
                 odom.pose.pose.position.z,
                 odom.pose.pose.orientation.x,
                 odom.pose.pose.orientation.y,
                 odom.pose.pose.orientation.z,
                 odom.pose.pose.orientation.w],
                dtype=np.float32,
            )
        else:
            odom_arr = np.zeros(7, dtype=np.float32)

        frame = {
            "timestamp": time.time(),
            "lidar_xyz": lidar_pts,
            "cmd_vel": cmd_vel,
            "odom": odom_arr,
            "lidar_point_count": len(lidar_pts),
        }
        self._frames.append(frame)
        self._frame_count += 1

    def _clearance_summary(self, lidar_pts):
        if len(lidar_pts) < 10:
            return "N/A"
        front_pts = lidar_pts[np.abs(np.arctan2(lidar_pts[:, 1], lidar_pts[:, 0])) < np.radians(18)]
        front_d = np.min(np.hypot(front_pts[:, 0], front_pts[:, 1])) if len(front_pts) > 0 else 4.5
        return f"front={front_d:.2f}m"

    def save(self):
        if not self._frames:
            rospy.logwarn("  没有采集到数据，跳过保存。")
            return

        elapsed = time.time() - self._start_time if self._start_time else 0

        metadata = {
            "collection_time_sec": round(elapsed, 1),
            "frame_count": len(self._frames),
            "output_path": self._output_path,
            "collection_rate_hz": self._rate,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }

        lidar_arrays = [f["lidar_xyz"] for f in self._frames]
        cmd_arrays = np.stack([f["cmd_vel"] for f in self._frames])
        odom_arrays = np.stack([f["odom"] for f in self._frames])
        timestamps = np.array([f["timestamp"] for f in self._frames], dtype=np.float64)
        point_counts = np.array([f["lidar_point_count"] for f in self._frames], dtype=np.int32)

        np.savez(
            self._output_path,
            lidar_points=np.array(lidar_arrays, dtype=object),
            cmd_vel=cmd_arrays,
            odom=odom_arrays,
            timestamps=timestamps,
            point_counts=point_counts,
            metadata=str(metadata),
        )
        file_size_mb = os.path.getsize(self._output_path) / (1024 * 1024)
        rospy.loginfo(
            "  数据已保存: %s (%.1f MB, %d 帧, %.1f 秒)",
            self._output_path, file_size_mb, len(self._frames), elapsed,
        )

    def run(self):
        rospy.loginfo("  数据采集器启动 — 按 Ctrl+C 停止并保存")
        rospy.loginfo("  输出文件: %s", os.path.abspath(self._output_path))
        rospy.loginfo("  采集频率: %d Hz", self._rate)

        rospy.sleep(1.0)
        self._start_time = time.time()
        rate = rospy.Rate(self._rate)
        last_log = time.time()

        while self._running and not rospy.is_shutdown():
            self._capture_frame()

            if time.time() - last_log >= 1.0:
                latest_pts = self._frames[-1]["lidar_xyz"] if self._frames else np.array([])
                summary = self._clearance_summary(latest_pts)
                sys.stdout.write(f"\r  已采集 {self._frame_count} 帧 | {summary}    ")
                sys.stdout.flush()
                last_log = time.time()

            if self._max_frames > 0 and self._frame_count >= self._max_frames:
                rospy.loginfo("  达到最大帧数限制 (%d)，停止采集。", self._max_frames)
                break

            rate.sleep()

        self.save()


def main():
    default_output = os.path.join(
        os.path.dirname(__file__), "..", "data",
        f"scene1_frames_{time.strftime('%Y%m%d_%H%M%S')}.npz",
    )

    parser = argparse.ArgumentParser(description="CRAIC 激光雷达数据采集器")
    parser.add_argument("--output", type=str, default=default_output,
                       help="输出 .npz 文件路径")
    parser.add_argument("--rate", type=int, default=10,
                       help="采集频率 (Hz)")
    parser.add_argument("--max-frames", type=int, default=0,
                       help="最大采集帧数 (0=无限制)")
    args = parser.parse_args()

    try:
        DataCollector(args.output, rate=args.rate, max_frames=args.max_frames).run()
    except rospy.ROSInterruptException:
        pass


if __name__ == "__main__":
    main()
