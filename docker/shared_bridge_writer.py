#!/usr/bin/env python3
"""
shared_bridge_writer.py — Runs inside the go2_sim container (ROS2 Jazzy).

Subscribes to Gazebo simulation topics and writes them as files to /shared/
so the behav container (ROS2 Humble) can read them without DDS cross-version issues.
"""

import json
import os
import struct
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan

SHARED_DIR = "/shared"
ODOM_FILE = os.path.join(SHARED_DIR, "odom.json")
ODOM_TMP = os.path.join(SHARED_DIR, "odom.json.tmp")
CAMERA_FILE = os.path.join(SHARED_DIR, "camera.bin")
CAMERA_TMP = os.path.join(SHARED_DIR, "camera.bin.tmp")
CAMERA_META_FILE = os.path.join(SHARED_DIR, "camera_meta.json")
CAMERA_META_TMP = os.path.join(SHARED_DIR, "camera_meta.json.tmp")
SCAN_FILE = os.path.join(SHARED_DIR, "scan.json")
SCAN_TMP = os.path.join(SHARED_DIR, "scan.json.tmp")

STARTUP_DELAY_S = 5


class SharedBridgeWriter(Node):
    def __init__(self):
        super().__init__("shared_bridge_writer")

        self.get_logger().info(
            f"Waiting {STARTUP_DELAY_S}s for Gazebo topics to become available..."
        )
        time.sleep(STARTUP_DELAY_S)

        os.makedirs(SHARED_DIR, exist_ok=True)

        self.odom_sub = self.create_subscription(
            Odometry,
            "/robot1/odom",
            self._odom_callback,
            10,
        )

        self.camera_sub = self.create_subscription(
            Image,
            "/robot1/camera/image_raw",
            self._camera_callback,
            10,
        )

        self.scan_sub = self.create_subscription(
            LaserScan,
            "/robot1/scan",
            self._scan_callback,
            10,
        )

        self.get_logger().info(
            "SharedBridgeWriter ready. Writing to /shared/odom.json, /shared/camera.bin, /shared/scan.json"
        )

    def _odom_callback(self, msg: Odometry) -> None:
        pose = msg.pose.pose
        twist = msg.twist.twist

        data = {
            "x": pose.position.x,
            "y": pose.position.y,
            "z": pose.position.z,
            "qx": pose.orientation.x,
            "qy": pose.orientation.y,
            "qz": pose.orientation.z,
            "qw": pose.orientation.w,
            "vx": twist.linear.x,
            "vy": twist.linear.y,
            "wz": twist.angular.z,
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "frame_id": msg.header.frame_id,
            "child_frame_id": msg.child_frame_id,
        }

        payload = json.dumps(data)
        with open(ODOM_TMP, "w") as f:
            f.write(payload)
        os.rename(ODOM_TMP, ODOM_FILE)

    def _camera_callback(self, msg: Image) -> None:
        # Write metadata atomically
        meta = {
            "encoding": msg.encoding,
            "width": msg.width,
            "height": msg.height,
        }
        meta_payload = json.dumps(meta)
        with open(CAMERA_META_TMP, "w") as f:
            f.write(meta_payload)
        os.rename(CAMERA_META_TMP, CAMERA_META_FILE)

        # Binary format: 3x uint32 little-endian header (width, height, step),
        # followed by raw image bytes.
        header = struct.pack("<III", msg.width, msg.height, msg.step)
        with open(CAMERA_TMP, "wb") as f:
            f.write(header)
            f.write(bytes(msg.data))
        os.rename(CAMERA_TMP, CAMERA_FILE)


    def _scan_callback(self, msg: LaserScan) -> None:
        data = {
            "angle_min": msg.angle_min,
            "angle_max": msg.angle_max,
            "angle_increment": msg.angle_increment,
            "time_increment": msg.time_increment,
            "scan_time": msg.scan_time,
            "range_min": msg.range_min,
            "range_max": msg.range_max,
            "ranges": list(msg.ranges),
            "intensities": list(msg.intensities),
            "stamp_sec": msg.header.stamp.sec,
            "stamp_nanosec": msg.header.stamp.nanosec,
            "frame_id": msg.header.frame_id,
        }
        with open(SCAN_TMP, "w") as f:
            f.write(json.dumps(data))
        os.rename(SCAN_TMP, SCAN_FILE)


def main() -> None:
    rclpy.init()
    node = SharedBridgeWriter()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
