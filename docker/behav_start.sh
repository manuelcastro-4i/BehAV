#!/bin/bash
source /opt/ros/humble/setup.bash
echo "[behav_start] Starting shared bridge reader..."
python3 /app/shared_bridge_reader.py &
READER_PID=$!
echo "[behav_start] Starting BehAV planner..."
exec "$@"
