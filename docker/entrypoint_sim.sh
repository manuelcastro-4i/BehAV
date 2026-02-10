#!/bin/bash
set -e

# Source ROS2 Jazzy setup
source /opt/ros/jazzy/setup.bash

# Source workspace if it exists
if [ -f /app/go2_sim/install/setup.bash ]; then
    source /app/go2_sim/install/setup.bash
fi

# Start Xvfb (virtual framebuffer) for headless rendering
echo "Starting Xvfb on display :99..."
Xvfb :99 -screen 0 1920x1080x24 &
XVFB_PID=$!
sleep 1

# Check if noVNC mode is enabled
if [ "${ENABLE_VNC}" = "true" ]; then
    echo "Starting noVNC mode (GUI accessible at http://localhost:6080)..."

    # Start lightweight window manager
    fluxbox &
    sleep 1

    # Start VNC server (no password, listening on port 5900)
    x11vnc -display :99 -forever -nopw -shared -rfbport 5900 &
    sleep 1

    # Start noVNC websocket proxy (port 6080 -> VNC port 5900)
    websockify --web /usr/share/novnc 6080 localhost:5900 &
    sleep 1

    echo "noVNC is ready at http://localhost:6080"
else
    echo "Running in headless mode (no GUI). Set ENABLE_VNC=true for browser-based GUI."
fi

# Execute the command (default: ros2 launch)
exec "$@"
