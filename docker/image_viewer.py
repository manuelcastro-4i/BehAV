#!/usr/bin/env python3
"""
BehAV Image Viewer — web UI for instructions, reset, and MJPEG streams.
Open http://localhost:8080 in any browser.
"""
import http.client
import json
import os
import queue
import socket
import threading
import time
import urllib.request

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# ── ROS2 frame state ──────────────────────────────────────────────────────────
_lock = threading.Lock()
_frames = {'costmap': None, 'trajectory': None}
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
        self.get_logger().info(
            'ImageViewer: subscribed to /behav_costmap and /traj_marked_image'
        )

    def _cb(self, msg, key):
        try:
            data = _ros_to_jpeg(msg)
            with _lock:
                _frames[key] = data
        except Exception as e:
            self.get_logger().warn(f'Frame error ({key}): {e}')


# ── SSE broadcast ─────────────────────────────────────────────────────────────
_sse_lock = threading.Lock()
_sse_queues: list = []


def broadcast(kind: str, text: str):
    """Send an SSE event to all connected clients."""
    data = f'event: {kind}\ndata: {json.dumps(text)}\n\n'.encode()
    with _sse_lock:
        dead = []
        for q in _sse_queues:
            try:
                q.put_nowait(data)
            except queue.Full:
                dead.append(q)
        for q in dead:
            _sse_queues.remove(q)


# ── Docker API helpers ────────────────────────────────────────────────────────
_DOCKER_SOCK = '/var/run/docker.sock'


class _DockerConn(http.client.HTTPConnection):
    """HTTPConnection tunnelled over the Docker Unix socket."""
    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.connect(_DOCKER_SOCK)


def _docker_post(path: str) -> int:
    """POST to Docker socket, return HTTP status code."""
    try:
        conn = _DockerConn('localhost')
        conn.request('POST', path, headers={'Content-Length': '0'})
        resp = conn.getresponse()
        resp.read()  # drain body
        return resp.status
    except Exception as e:
        print(f'[Docker] POST {path}: {e}')
        return 0


def _docker_get(path: str) -> tuple:
    """GET from Docker socket, return (status_code, parsed_json)."""
    try:
        conn = _DockerConn('localhost')
        conn.request('GET', path)
        resp = conn.getresponse()
        body = resp.read()
        return resp.status, json.loads(body)
    except Exception as e:
        print(f'[Docker] GET {path}: {e}')
        return 0, {}


# ── Reset logic ───────────────────────────────────────────────────────────────
def _reset_stack():
    """Restart all containers in order; broadcast progress via SSE."""
    broadcast('system', 'Restarting simulator...')
    status = _docker_post('/containers/go2_simulator/restart?t=5')
    if status not in (200, 204):
        broadcast('system',
                  f'Simulator restart failed (HTTP {status}). '
                  'Is the Docker socket mounted?')
        return

    broadcast('system', 'Waiting for simulator to become healthy...')
    for _ in range(30):   # up to 90 s
        time.sleep(3)
        status, data = _docker_get('/containers/go2_simulator/json')
        if status == 200:
            health = data.get('State', {}).get('Health', {}).get('Status', '')
            if health == 'healthy':
                broadcast('system',
                          'Simulator is healthy. Restarting other services...')
                break
    else:
        broadcast('system',
                  'Timeout waiting for simulator. Proceeding anyway...')

    time.sleep(2)
    _docker_post('/containers/behav_bridge_reader/restart?t=5')
    broadcast('system', 'Bridge reader restarted.')

    time.sleep(2)
    _docker_post('/containers/behav_planner/restart?t=5')
    broadcast('system', 'Planner restarted.')

    time.sleep(2)
    _docker_post('/containers/behav_director/restart?t=5')
    broadcast('system', 'Director restarted. Enter a new instruction.')


# ── Director API ──────────────────────────────────────────────────────────────
_DIRECTOR_URL = os.environ.get('DIRECTOR_API_URL', 'http://director:8889')


def _send_instruction(instruction: str):
    """Forward instruction to director; broadcast result via SSE."""
    broadcast('system', 'Processing...')
    try:
        body = json.dumps({'instruction': instruction}).encode()
        req = urllib.request.Request(
            f'{_DIRECTOR_URL}/instruction',
            data=body,
            headers={'Content-Type': 'application/json'},
            method='POST',
        )
        with urllib.request.urlopen(req, timeout=60) as resp:
            result = json.loads(resp.read())
        prompts = result.get('prompts', [])
        broadcast('system',
                  f'Published to robot. Prompts: {", ".join(str(p) for p in prompts)}')
    except Exception as e:
        broadcast('system', f'Error: {e}')


# ── Waiting frames ────────────────────────────────────────────────────────────
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


# ── HTML ──────────────────────────────────────────────────────────────────────
HTML = """<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  <title>BehAV Robot Interface</title>
  <style>
    * { box-sizing: border-box; margin: 0; padding: 0; }
    body { background: #111; color: #eee; font-family: sans-serif;
           height: 100vh; display: flex; flex-direction: column; overflow: hidden; }
    header { background: #1a1a2e; padding: 10px 18px; display: flex;
             align-items: center; gap: 12px; border-bottom: 1px solid #333; flex-shrink: 0; }
    header h1 { font-size: 17px; color: #4af; }
    header span { color: #555; font-size: 12px; }
    .main { display: flex; flex: 1; overflow: hidden; }

    /* ── Left panel: chat ── */
    .chat-panel { width: 33%; min-width: 270px; display: flex;
                  flex-direction: column; border-right: 1px solid #2a2a2a;
                  padding: 10px; gap: 8px; }
    .chat-panel h2 { font-size: 11px; color: #777; text-transform: uppercase;
                     letter-spacing: 1px; }
    .messages { flex: 1; overflow-y: auto; background: #181818;
                border-radius: 6px; padding: 8px; font-size: 12px;
                display: flex; flex-direction: column; gap: 5px; }
    .msg { padding: 5px 8px; border-radius: 4px; line-height: 1.45; word-break: break-word; }
    .msg.user   { background: #173050; color: #bde; align-self: flex-end; max-width: 92%; }
    .msg.system { background: #162416; color: #8c8; align-self: flex-start; max-width: 92%; }
    .msg.robot  { background: #251625; color: #c8c; align-self: flex-start; max-width: 92%; }
    .msg.info   { color: #555; font-style: italic; align-self: center; font-size: 11px; }
    .input-row { display: flex; gap: 6px; }
    textarea { flex: 1; background: #1e1e1e; border: 1px solid #3a3a3a; color: #ddd;
               border-radius: 6px; padding: 7px 9px; font-size: 12px; resize: none;
               height: 56px; font-family: inherit; }
    textarea:focus { outline: none; border-color: #4af; }
    .btn-send { background: #1a5070; border: none; color: #cef; border-radius: 6px;
                padding: 0 12px; cursor: pointer; font-size: 12px; white-space: nowrap; }
    .btn-send:hover { background: #206090; }
    .btn-send:disabled { background: #2a2a2a; color: #555; cursor: default; }
    .btn-reset { width: 100%; padding: 9px; background: #5a1515; border: none;
                 color: #faa; border-radius: 6px; cursor: pointer; font-size: 12px;
                 font-weight: bold; letter-spacing: 0.5px; }
    .btn-reset:hover { background: #7a1c1c; }
    .btn-reset:disabled { background: #2a2a2a; color: #555; cursor: default; }

    /* ── Right panel: streams ── */
    .stream-panel { flex: 1; display: flex; flex-direction: column;
                    padding: 10px; gap: 10px; overflow: hidden; }
    .stream-row { display: flex; gap: 10px; flex: 1; overflow: hidden; }
    .box { flex: 1; background: #1c1c1c; border-radius: 8px; padding: 10px;
           display: flex; flex-direction: column; gap: 6px; overflow: hidden; }
    .box h2 { font-size: 11px; color: #777; text-transform: uppercase;
              letter-spacing: 1px; flex-shrink: 0; }
    .box img { width: 100%; height: 100%; object-fit: contain; border-radius: 4px; }
  </style>
</head>
<body>
  <header>
    <h1>BehAV Robot Interface</h1>
    <span>Live navigation control</span>
  </header>
  <div class="main">

    <!-- Left: console -->
    <div class="chat-panel">
      <h2>Navigation Console</h2>
      <div class="messages" id="messages">
        <div class="msg info">Waiting for instruction...</div>
      </div>
      <div class="input-row">
        <textarea id="instruction"
          placeholder="e.g. Stay on the pavement, avoid the grass, and go to the red marker at 17 meters"
        ></textarea>
        <button class="btn-send" id="btn-send" onclick="sendInstruction()">Send</button>
      </div>
      <button class="btn-reset" id="btn-reset" onclick="resetStack()">
        &#x21BA;&nbsp; Reset Stack
      </button>
    </div>

    <!-- Right: streams -->
    <div class="stream-panel">
      <div class="stream-row">
        <div class="box">
          <h2>Behavioral Cost Map (CLIPSeg)</h2>
          <img src="/stream/costmap" alt="costmap">
        </div>
        <div class="box">
          <h2>Planned Trajectory</h2>
          <img src="/stream/trajectory" alt="trajectory">
        </div>
      </div>
    </div>

  </div>
  <script>
    // ── SSE ──────────────────────────────────────────────────────────────────
    const evtSrc = new EventSource('/events');
    evtSrc.addEventListener('system', e => addMsg('system', JSON.parse(e.data)));
    evtSrc.addEventListener('robot',  e => addMsg('robot',  JSON.parse(e.data)));
    evtSrc.onerror = () => addMsg('info', 'SSE reconnecting...');

    function addMsg(cls, text) {
      const div = document.createElement('div');
      div.className = 'msg ' + cls;
      div.textContent = text;
      const box = document.getElementById('messages');
      box.appendChild(div);
      box.scrollTop = box.scrollHeight;
    }

    // ── Actions ──────────────────────────────────────────────────────────────
    function sendInstruction() {
      const inp  = document.getElementById('instruction');
      const text = inp.value.trim();
      if (!text) return;
      addMsg('user', text);
      inp.value = '';
      setDisabled(true);
      fetch('/send', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({instruction: text}),
      })
        .catch(e => addMsg('system', 'Send error: ' + e))
        .finally(() => setDisabled(false));
    }

    function resetStack() {
      if (!confirm('Reset the entire robot stack?\\nThis will restart all containers.')) return;
      setDisabled(true);
      fetch('/reset', {method: 'POST'})
        .catch(e => addMsg('system', 'Reset error: ' + e))
        .finally(() => setDisabled(false));
    }

    function setDisabled(v) {
      document.getElementById('btn-send').disabled  = v;
      document.getElementById('btn-reset').disabled = v;
    }

    // Enter (without Shift) submits
    document.getElementById('instruction').addEventListener('keydown', e => {
      if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendInstruction(); }
    });
  </script>
</body>
</html>
""".encode('utf-8')


# ── HTTP handler ──────────────────────────────────────────────────────────────
class Handler(BaseHTTPRequestHandler):
    def log_message(self, *_):
        pass  # suppress per-request logs

    # ── GET ──────────────────────────────────────────────────────────────────
    def do_GET(self):
        if self.path == '/':
            self._respond(200, 'text/html; charset=utf-8', HTML)
        elif self.path == '/events':
            self._handle_sse()
        elif self.path.startswith('/stream/'):
            self._handle_stream()
        else:
            self.send_error(404)

    # ── POST ─────────────────────────────────────────────────────────────────
    def do_POST(self):
        if self.path == '/send':
            data = self._read_json()
            instruction = (data or {}).get('instruction', '').strip()
            if not instruction:
                self._json(400, {'error': 'instruction required'})
                return
            self._json(200, {'status': 'processing'})
            threading.Thread(
                target=_send_instruction, args=(instruction,), daemon=True
            ).start()

        elif self.path == '/reset':
            self._json(200, {'status': 'resetting'})
            threading.Thread(target=_reset_stack, daemon=True).start()

        elif self.path == '/notify':
            data = self._read_json()
            text = (data or {}).get('text', '')
            broadcast('robot', str(text))
            self._json(200, {'status': 'ok'})

        else:
            self.send_error(404)

    # ── SSE stream ────────────────────────────────────────────────────────────
    def _handle_sse(self):
        self.send_response(200)
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('X-Accel-Buffering', 'no')
        self.end_headers()
        q: queue.Queue = queue.Queue(maxsize=100)
        with _sse_lock:
            _sse_queues.append(q)
        try:
            while True:
                try:
                    data = q.get(timeout=25)
                    self.wfile.write(data)
                    self.wfile.flush()
                except queue.Empty:
                    self.wfile.write(b': keepalive\n\n')
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            with _sse_lock:
                if q in _sse_queues:
                    _sse_queues.remove(q)

    # ── MJPEG stream ──────────────────────────────────────────────────────────
    def _handle_stream(self):
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
                time.sleep(0.1)   # ~10 fps
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _read_json(self):
        try:
            length = int(self.headers.get('Content-Length', 0))
            return json.loads(self.rfile.read(length))
        except Exception:
            return None

    def _respond(self, code: int, content_type: str, body: bytes):
        self.send_response(code)
        self.send_header('Content-Type', content_type)
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()

    def _json(self, code: int, data: dict):
        self._respond(code, 'application/json', json.dumps(data).encode())


# ── HTTP server ───────────────────────────────────────────────────────────────
def _run_http(port: int = 8080):
    server = ThreadingHTTPServer(('0.0.0.0', port), Handler)
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
