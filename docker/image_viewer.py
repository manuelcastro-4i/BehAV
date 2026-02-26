#!/usr/bin/env python3
"""
BehAV Image Viewer — serves costmap and trajectory images as MJPEG streams.
Open http://localhost:8080 in any browser.
"""
import io
import threading
import time

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from http.server import BaseHTTPRequestHandler, HTTPServer

# Latest frames (thread-safe via lock)
_lock = threading.Lock()
_frames = {
    'costmap': None,
    'trajectory': None,
}
_bridge = CvBridge()


def _ros_to_jpeg(msg: Image) -> bytes:
    img = _bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
    _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return buf.tobytes()


class ImageViewerNode(Node):
    def __init__(self):
        super().__init__('behav_image_viewer')
        self.create_subscription(Image, '/behav_costmap',
                                 lambda m: self._cb(m, 'costmap'), 5)
        self.create_subscription(Image, '/traj_marked_image',
                                 lambda m: self._cb(m, 'trajectory'), 5)
        self.get_logger().info('ImageViewer: subscribed to /behav_costmap and /traj_marked_image')

    def _cb(self, msg, key):
        try:
            data = _ros_to_jpeg(msg)
            with _lock:
                _frames[key] = data
        except Exception as e:
            self.get_logger().warn(f'Frame error ({key}): {e}')


HTML = b"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BehAV Viewer</title>
  <style>
    body { background:#111; color:#eee; font-family:sans-serif; text-align:center; margin:0; padding:20px; }
    h1 { color:#4af; }
    .grid { display:flex; gap:20px; justify-content:center; flex-wrap:wrap; margin-top:20px; }
    .box { background:#222; border-radius:8px; padding:12px; }
    .box h2 { margin:0 0 8px; font-size:14px; color:#aaa; }
    img { max-width:640px; width:100%; border-radius:4px; }
  </style>
</head>
<body>
  <h1>BehAV Live Viewer</h1>
  <div class="grid">
    <div class="box">
      <h2>Behavioral Cost Map (CLIPSeg)</h2>
      <img src="/stream/costmap">
    </div>
    <div class="box">
      <h2>Planned Trajectory</h2>
      <img src="/stream/trajectory">
    </div>
  </div>
  <p style="color:#555;font-size:12px">Auto-refreshing MJPEG streams</p>
</body>
</html>
"""

WAITING_FRAME: bytes = b''


def _make_waiting_jpeg(text: str) -> bytes:
    img = np.zeros((240, 320, 3), dtype=np.uint8)
    cv2.putText(img, 'Waiting...', (60, 100), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, (100, 100, 100), 2)
    cv2.putText(img, text, (10, 140), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (80, 80, 80), 1)
    _, buf = cv2.imencode('.jpg', img)
    return buf.tobytes()


_WAITING = {
    'costmap':    _make_waiting_jpeg('/behav_costmap'),
    'trajectory': _make_waiting_jpeg('/traj_marked_image'),
}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress request logs

    def do_GET(self):
        if self.path == '/':
            self.send_response(200)
            self.send_header('Content-Type', 'text/html')
            self.end_headers()
            self.wfile.write(HTML)

        elif self.path.startswith('/stream/'):
            key = self.path.split('/')[-1]
            if key not in _frames:
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type',
                             'multipart/x-mixed-replace; boundary=frame')
            self.end_headers()
            try:
                while True:
                    with _lock:
                        frame = _frames.get(key) or _WAITING[key]
                    self.wfile.write(
                        b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' +
                        frame + b'\r\n'
                    )
                    self.wfile.flush()
                    time.sleep(0.1)  # ~10 fps
            except (BrokenPipeError, ConnectionResetError):
                pass
        else:
            self.send_error(404)


def _run_http(port: int = 8080):
    server = HTTPServer(('0.0.0.0', port), Handler)
    print(f'[image_viewer] HTTP server at http://localhost:{port}')
    server.serve_forever()


def main():
    rclpy.init()
    node = ImageViewerNode()

    http_thread = threading.Thread(target=_run_http, daemon=True)
    http_thread.start()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
