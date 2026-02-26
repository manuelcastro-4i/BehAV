#!/usr/bin/env python3
"""
shared_bridge_reader.py — Runs inside the behav container (ROS2 Humble).

Reads odometry and camera data written by shared_bridge_writer.py in the
go2_sim container from the /shared/ volume and republishes them as ROS2
topics, bypassing cross-version DDS incompatibility.
"""

import json
import os
import struct
import time

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image, LaserScan
from builtin_interfaces.msg import Time
from std_msgs.msg import Header
from geometry_msgs.msg import (
    Pose,
    Twist,
    PoseWithCovariance,
    TwistWithCovariance,
)

SHARED_DIR = "/shared"
ODOM_FILE = os.path.join(SHARED_DIR, "odom.json")
CAMERA_FILE = os.path.join(SHARED_DIR, "camera.bin")
CAMERA_META_FILE = os.path.join(SHARED_DIR, "camera_meta.json")
SCAN_FILE = os.path.join(SHARED_DIR, "scan.json")
CMD_VEL_FILE = os.path.join(SHARED_DIR, "cmd_vel.json")
CMD_VEL_TMP = os.path.join(SHARED_DIR, "cmd_vel.json.tmp")

ODOM_RATE_HZ = 50.0
CAMERA_RATE_HZ = 10.0
SCAN_RATE_HZ = 10.0


class SharedBridgeReader(Node):
    def __init__(self):
        super().__init__("shared_bridge_reader")

        self.odom_pub = self.create_publisher(Odometry, "/robot1/odom", 10)
        self.camera_pub = self.create_publisher(
            Image, "/robot1/camera/image_raw", 10
        )
        self.scan_pub = self.create_publisher(LaserScan, "/robot1/scan", 10)

        self._odom_mtime: float = 0.0
        self._camera_mtime: float = 0.0
        self._scan_mtime: float = 0.0

        self.create_timer(1.0 / ODOM_RATE_HZ, self._publish_odom)
        self.create_timer(1.0 / CAMERA_RATE_HZ, self._publish_camera)
        self.create_timer(1.0 / SCAN_RATE_HZ, self._publish_scan)

        # cmd_vel relay: subscribe on Humble, write to /shared/cmd_vel.json for go2_sim
        from geometry_msgs.msg import Twist as TwistMsg
        self.create_subscription(TwistMsg, "/robot1/cmd_vel", self._cmd_vel_callback, 10)

        self.get_logger().info(
            "SharedBridgeReader ready. Reading from /shared/ and publishing odom+camera+scan to ROS2. "
            "Also relaying /robot1/cmd_vel → /shared/cmd_vel.json."
        )

    def _publish_odom(self) -> None:
        if not os.path.exists(ODOM_FILE):
            return

        try:
            mtime = os.stat(ODOM_FILE).st_mtime
        except OSError:
            return

        if mtime <= self._odom_mtime:
            return
        self._odom_mtime = mtime

        try:
            with open(ODOM_FILE, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        msg = Odometry()

        stamp = Time()
        stamp.sec = int(data.get("stamp_sec", 0))
        stamp.nanosec = int(data.get("stamp_nanosec", 0))

        header = Header()
        header.stamp = stamp
        header.frame_id = data.get("frame_id", "odom")

        msg.header = header
        msg.child_frame_id = data.get("child_frame_id", "base_link")

        pose = Pose()
        pose.position.x = float(data.get("x", 0.0))
        pose.position.y = float(data.get("y", 0.0))
        pose.position.z = float(data.get("z", 0.0))
        pose.orientation.x = float(data.get("qx", 0.0))
        pose.orientation.y = float(data.get("qy", 0.0))
        pose.orientation.z = float(data.get("qz", 0.0))
        pose.orientation.w = float(data.get("qw", 1.0))

        pose_with_cov = PoseWithCovariance()
        pose_with_cov.pose = pose
        msg.pose = pose_with_cov

        twist = Twist()
        twist.linear.x = float(data.get("vx", 0.0))
        twist.linear.y = float(data.get("vy", 0.0))
        twist.angular.z = float(data.get("wz", 0.0))

        twist_with_cov = TwistWithCovariance()
        twist_with_cov.twist = twist
        msg.twist = twist_with_cov

        self.odom_pub.publish(msg)

    def _publish_camera(self) -> None:
        if not os.path.exists(CAMERA_FILE) or not os.path.exists(CAMERA_META_FILE):
            return

        try:
            mtime = os.stat(CAMERA_FILE).st_mtime
        except OSError:
            return

        if mtime <= self._camera_mtime:
            return
        self._camera_mtime = mtime

        try:
            with open(CAMERA_META_FILE, "r") as f:
                meta = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        try:
            with open(CAMERA_FILE, "rb") as f:
                raw = f.read()
        except OSError:
            return

        # Parse header: 3x uint32 little-endian (width, height, step)
        header_size = struct.calcsize("<III")
        if len(raw) < header_size:
            return

        width, height, step = struct.unpack_from("<III", raw, 0)
        image_data = raw[header_size:]

        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_link"
        msg.width = width
        msg.height = height
        msg.step = step
        msg.encoding = meta.get("encoding", "rgb8")
        msg.is_bigendian = False
        msg.data = list(image_data)

        self.camera_pub.publish(msg)


    def _publish_scan(self) -> None:
        if not os.path.exists(SCAN_FILE):
            return

        try:
            mtime = os.stat(SCAN_FILE).st_mtime
        except OSError:
            return

        if mtime <= self._scan_mtime:
            return
        self._scan_mtime = mtime

        try:
            with open(SCAN_FILE, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        msg = LaserScan()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = data.get("frame_id", "lidar")
        msg.angle_min = float(data.get("angle_min", 0.0))
        msg.angle_max = float(data.get("angle_max", 0.0))
        msg.angle_increment = float(data.get("angle_increment", 0.0))
        msg.time_increment = float(data.get("time_increment", 0.0))
        msg.scan_time = float(data.get("scan_time", 0.0))
        msg.range_min = float(data.get("range_min", 0.0))
        msg.range_max = float(data.get("range_max", 0.0))
        msg.ranges = [float(r) for r in data.get("ranges", [])]
        msg.intensities = [float(i) for i in data.get("intensities", [])]

        self.scan_pub.publish(msg)


    def _cmd_vel_callback(self, msg) -> None:
        """Write cmd_vel from Humble /robot1/cmd_vel → /shared/cmd_vel.json for go2_sim relay."""
        data = {
            "linear_x": msg.linear.x,
            "linear_y": msg.linear.y,
            "angular_z": msg.angular.z,
            "stamp": time.time(),
        }
        try:
            with open(CMD_VEL_TMP, "w") as f:
                f.write(json.dumps(data))
            os.rename(CMD_VEL_TMP, CMD_VEL_FILE)
        except OSError:
            pass


def main() -> None:
    rclpy.init()
    node = SharedBridgeReader()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
