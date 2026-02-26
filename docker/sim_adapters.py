#!/usr/bin/env python3
"""
Sim Adapters - Bridges missing topics for the go2_ros2_sim_py simulation.

Starts a ros_gz_bridge subprocess that bridges:
  /robot1/odom            (nav_msgs/Odometry)   - from OdometryPublisher plugin
  /robot1/camera/image_raw (sensor_msgs/Image)  - from camera_face sensor
  /world_robot_poses       (PoseArray)           - fallback odom source

Also runs a TF-listener fallback: if the odom bridge doesn't deliver data
(e.g. OdometryPublisher plugin not loaded) we derive odom from TF or PoseArray.
"""

import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import rclpy
import rclpy.time
from geometry_msgs.msg import PoseArray, Twist
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformListener

CMD_VEL_FILE = "/shared/cmd_vel.json"
CMD_VEL_MAX_AGE_S = 0.5   # stop if no fresh cmd_vel within this window
CMD_VEL_STARTUP_DELAY_S = 20.0  # don't relay cmd_vel until Gazebo physics has settled

# ── Bridge process ──────────────────────────────────────────────────────────
BRIDGE_YAML = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'gz_pose_bridge.yaml')
# Fallback path for when mounted at container root
_BRIDGE_YAML_CONTAINER = '/app/gz_pose_bridge.yaml'


def start_pose_bridge() -> subprocess.Popen:
    """Launch ros_gz_bridge for world dynamic poses in background."""
    yaml_path = BRIDGE_YAML if os.path.exists(BRIDGE_YAML) else _BRIDGE_YAML_CONTAINER
    # In ROS2 Jazzy the executable is 'parameter_bridge' (not 'ros_gz_bridge')
    cmd = [
        'ros2', 'run', 'ros_gz_bridge', 'parameter_bridge',
        '--ros-args', '-p', f'config_file:={yaml_path}'
    ]
    print(f"[sim_adapters] Starting pose bridge: {' '.join(cmd)}")
    return subprocess.Popen(cmd)


# ── Synthetic camera image ───────────────────────────────────────────────────
def make_synthetic_image(w: int = 640, h: int = 480) -> Image:
    """
    Simple synthetic image that gives CLIPSeg something reasonable to segment:
      top 40%  - greenish sky/vegetation
      mid 20%  - transition
      bottom 40% - gray pavement
    """
    data = np.zeros((h, w, 3), dtype=np.uint8)
    split = int(h * 0.4)
    data[:split, :]       = [70, 130, 60]   # vegetation / sky
    data[split:split*2, :] = [100, 110, 90] # transition
    data[split*2:, :]     = [150, 150, 145] # pavement

    msg = Image()
    msg.width = w
    msg.height = h
    msg.encoding = 'rgb8'
    msg.step = w * 3
    msg.is_bigendian = False
    msg.data = data.tobytes()
    return msg


# ── ROS2 node ────────────────────────────────────────────────────────────────
class SimAdapters(Node):

    # TF candidate frames to search for robot pose
    _TF_PARENTS  = ['world', 'odom', 'map']
    _TF_CHILDREN = [
        'go2/trunk', 'trunk', 'base_link', 'go2/base',
        'base', 'robot_base', 'go2',
    ]

    def __init__(self):
        super().__init__('sim_adapters')

        # Fallback odometry publisher (only used when bridge odom is absent)
        self._odom_pub = self.create_publisher(Odometry, '/robot1/odom', 10)

        # TF listener (secondary fallback)
        self._tf_buf = Buffer()
        self._tf_listener = TransformListener(self._tf_buf, self)
        self._tf_found_frames: tuple | None = None

        # Monitor bridge odom: if we receive from the bridge, stop the fallback
        self._bridge_odom_alive = False
        self.create_subscription(
            Odometry, '/robot1/odom', self._bridge_odom_cb, 10
        )

        # PoseArray subscriber (last-resort fallback)
        self.create_subscription(
            PoseArray, '/world_robot_poses', self._pose_array_cb, 10
        )

        # State for velocity computation (used by TF/PoseArray fallbacks)
        self._prev_t   = self.get_clock().now()
        self._prev_x   = 0.0
        self._prev_y   = 0.0
        self._prev_yaw = 0.0

        # Fallback odom timer: only fires when bridge odom is not flowing
        self.create_timer(0.02, self._tf_odom_tick)   # 50 Hz

        # cmd_vel relay: read /shared/cmd_vel.json → publish on Jazzy domain
        # (ros_gz_bridge then forwards to Gazebo → VelocityControl plugin moves robot)
        self._cmd_vel_pub = self.create_publisher(Twist, '/robot1/cmd_vel', 10)
        self._cmd_vel_ready = False  # wait for Gazebo to settle before relaying
        self._cmd_vel_start_time = time.time()
        self.create_timer(0.05, self._cmd_vel_tick)   # 20 Hz

        self.get_logger().info(
            "SimAdapters started — monitoring /robot1/odom bridge; "
            "TF/PoseArray fallback active if bridge silent; "
            "cmd_vel relay /shared/cmd_vel.json → /robot1/cmd_vel active"
        )

    # ── cmd_vel relay: shared file → Jazzy ROS2 → Gazebo ────────────────────

    def _cmd_vel_tick(self):
        """Read /shared/cmd_vel.json and publish to /robot1/cmd_vel on Jazzy domain."""
        # Wait until Gazebo physics has settled before relaying any commands
        if not self._cmd_vel_ready:
            if time.time() - self._cmd_vel_start_time < CMD_VEL_STARTUP_DELAY_S:
                return
            self._cmd_vel_ready = True
            self.get_logger().info("cmd_vel relay active — Gazebo physics settled")

        if not os.path.exists(CMD_VEL_FILE):
            return
        try:
            with open(CMD_VEL_FILE, "r") as f:
                data = json.load(f)
        except (OSError, json.JSONDecodeError):
            return

        # Safety: ignore stale commands (planner may have stopped)
        age = time.time() - data.get("stamp", 0.0)
        if age > CMD_VEL_MAX_AGE_S:
            return

        msg = Twist()
        msg.linear.x = float(data.get("linear_x", 0.0))
        msg.linear.y = float(data.get("linear_y", 0.0))
        msg.angular.z = float(data.get("angular_z", 0.0))
        self._cmd_vel_pub.publish(msg)

    # ── Bridge odom watchdog ─────────────────────────────────────────────────

    def _bridge_odom_cb(self, _msg: Odometry):
        """Detect that the ros_gz_bridge is delivering real odom."""
        if not self._bridge_odom_alive:
            self.get_logger().info(
                "SimAdapters: bridge odom detected — disabling TF/PoseArray fallback"
            )
        self._bridge_odom_alive = True

    # ── Odom from TF ────────────────────────────────────────────────────────

    def _tf_odom_tick(self):
        # Bridge is alive — no need for fallback
        if self._bridge_odom_alive:
            return

        # Once we've found the right frames, use them directly
        if self._tf_found_frames:
            parent, child = self._tf_found_frames
            try:
                t = self._tf_buf.lookup_transform(parent, child, rclpy.time.Time())
                self._pub_odom_from_transform(t)
                return
            except Exception:
                self._tf_found_frames = None  # lost the transform, re-probe

        # Probe for available TF frames
        for parent in self._TF_PARENTS:
            for child in self._TF_CHILDREN:
                try:
                    t = self._tf_buf.lookup_transform(parent, child, rclpy.time.Time())
                    self._tf_found_frames = (parent, child)
                    self.get_logger().info(
                        f"SimAdapters: using TF {parent} → {child} for odom"
                    )
                    self._pub_odom_from_transform(t)
                    return
                except Exception:
                    pass

    def _pub_odom_from_transform(self, t):
        now = self.get_clock().now()
        dt  = (now - self._prev_t).nanoseconds / 1e9
        if dt < 1e-4:
            return

        x = t.transform.translation.x
        y = t.transform.translation.y
        qx, qy, qz, qw = (
            t.transform.rotation.x, t.transform.rotation.y,
            t.transform.rotation.z, t.transform.rotation.w,
        )
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = t.transform.translation.z
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x    = (x - self._prev_x) / dt
        odom.twist.twist.linear.y    = (y - self._prev_y) / dt
        odom.twist.twist.angular.z   = (yaw - self._prev_yaw) / dt

        self._odom_pub.publish(odom)
        self._odom_published = True
        self._prev_x, self._prev_y, self._prev_yaw = x, y, yaw
        self._prev_t = now

    # ── Odom from PoseArray (fallback) ──────────────────────────────────────

    def _pose_array_cb(self, msg: PoseArray):
        """Last-resort fallback: derive odom from Gazebo world poses."""
        if self._bridge_odom_alive or self._tf_found_frames:
            return
        if not msg.poses:
            return

        now = self.get_clock().now()
        dt  = (now - self._prev_t).nanoseconds / 1e9
        if dt < 1e-4:
            return

        # In an empty world the robot is the only dynamic entity.
        # Average all poses to get the robot centroid (avoids picking a leg).
        xs   = [p.position.x for p in msg.poses]
        ys   = [p.position.y for p in msg.poses]
        zs   = [p.position.z for p in msg.poses]
        # Use the pose with maximum Z (trunk is highest link on a legged robot)
        idx  = int(np.argmax(zs))
        p    = msg.poses[idx]

        x  = float(np.mean(xs))
        y  = float(np.mean(ys))
        qx, qy, qz, qw = (
            p.orientation.x, p.orientation.y,
            p.orientation.z, p.orientation.w,
        )
        yaw = math.atan2(2*(qw*qz + qx*qy), 1 - 2*(qy*qy + qz*qz))

        odom = Odometry()
        odom.header.stamp    = now.to_msg()
        odom.header.frame_id = 'odom'
        odom.child_frame_id  = 'base_link'
        odom.pose.pose.position.x    = x
        odom.pose.pose.position.y    = y
        odom.pose.pose.position.z    = float(np.mean(zs))
        odom.pose.pose.orientation.x = qx
        odom.pose.pose.orientation.y = qy
        odom.pose.pose.orientation.z = qz
        odom.pose.pose.orientation.w = qw
        odom.twist.twist.linear.x    = (x - self._prev_x) / dt
        odom.twist.twist.linear.y    = (y - self._prev_y) / dt
        odom.twist.twist.angular.z   = (yaw - self._prev_yaw) / dt

        self._odom_pub.publish(odom)
        self._prev_x, self._prev_y, self._prev_yaw = x, y, yaw
        self._prev_t = now


# ── Entry point ──────────────────────────────────────────────────────────────

def main():
    # Start the pose bridge in background (for PoseArray fallback)
    bridge_proc = start_pose_bridge()

    rclpy.init()
    node = SimAdapters()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()
        bridge_proc.terminate()


if __name__ == '__main__':
    main()
