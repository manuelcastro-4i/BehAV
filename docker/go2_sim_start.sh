#!/bin/bash
# go2_sim_start.sh — Starts the Go2 Gazebo simulation plus the sim_adapters
# node that provides /robot1/odom and /robot1/camera/image_raw.
set -e

echo "[go2_sim_start] Launching Gazebo simulation..."
ros2 launch gazebo_sim launch_sim.launch.py &
SIM_PID=$!

echo "[go2_sim_start] Waiting 15s for Gazebo to initialize..."
sleep 15

echo "[go2_sim_start] Starting sim_adapters (odom + camera bridge)..."
python3 /app/sim_adapters.py &
ADAPTERS_PID=$!

echo "[go2_sim_start] Starting shared bridge writer..."
python3 /app/shared_bridge_writer.py &
WRITER_PID=$!

echo "[go2_sim_start] All processes running. SIM_PID=$SIM_PID ADAPTERS_PID=$ADAPTERS_PID WRITER_PID=$WRITER_PID"

# Forward SIGTERM to children
trap "kill $SIM_PID $ADAPTERS_PID $WRITER_PID 2>/dev/null" SIGTERM SIGINT

# Wait for the sim (main process)
wait $SIM_PID
