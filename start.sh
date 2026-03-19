#!/usr/bin/env bash
# ============================================
#  BehAV - Start Services
# ============================================
set -euo pipefail

cd "$(dirname "$0")"

# ── OS detection ──────────────────────────────────────────────────────────────
# Native Linux: opens Gazebo + RViz windows directly via X11 passthrough.
# Windows / WSL: uses noVNC browser interface (http://localhost:6080).
if [[ "$(uname -s)" == "Linux" ]] \
    && [[ -z "${WSL_DISTRO_NAME:-}" ]] \
    && ! grep -qi "microsoft" /proc/version 2>/dev/null; then
  NATIVE_LINUX=true
else
  NATIVE_LINUX=false
fi

echo
echo "============================================"
echo "  BehAV - Starting Services"
echo "============================================"
echo

# Remove stale go2_sim / bridge_reader containers so Xvfb doesn't hit a lock
# file left over from a previous run. Other containers are recreated only if
# their definition changed.
echo "Removing stale sim containers (prevents Xvfb /tmp/.X99-lock issues)..."
docker compose rm -f go2_sim bridge_reader 2>/dev/null || true

# Clear stale cmd_vel.json so the robot doesn't shoot off before Gazebo settles
echo "Clearing shared volume..."
docker run --rm -v behav_shared_bridge:/shared alpine \
  sh -c "rm -f /shared/cmd_vel.json" 2>/dev/null || true

echo

if [[ "$NATIVE_LINUX" == "true" ]]; then
  echo "Detected native Linux — using X11 passthrough for Gazebo and RViz..."
  # Allow Docker containers to open windows on the local X server
  xhost +local:docker >/dev/null 2>&1 || true
  export LIBGL_ALWAYS_SOFTWARE=0
  export ENABLE_VNC=false

  echo "Starting all BehAV services..."
  docker compose up -d go2_sim bridge_reader behav director viewer landmark_detector rviz2
else
  echo "Detected Windows / WSL — using noVNC browser interface..."
  export ENABLE_VNC=true

  echo "Starting all BehAV services..."
  docker compose up -d go2_sim bridge_reader behav director viewer landmark_detector
fi

echo
echo "Waiting for go2_sim to be healthy (Gazebo takes ~30 s)..."
for i in $(seq 1 30); do
  STATUS=$(docker inspect --format='{{.State.Health.Status}}' go2_simulator 2>/dev/null || echo "unknown")
  if [[ "$STATUS" == "healthy" ]]; then
    echo "  go2_sim is healthy!"
    break
  fi
  printf "  [%2d/30] %s...\r" "$i" "$STATUS"
  sleep 2
done
echo

echo "============================================"
echo "  Services Started!"
echo "============================================"
echo

if [[ "$NATIVE_LINUX" == "true" ]]; then
  echo "  Gazebo and RViz windows will open on your desktop."
else
  echo "  Gazebo GUI (noVNC):    http://localhost:6080"
fi

echo
echo "  Robot Interface:       http://localhost:8080"
echo "  (Type navigation instructions in the web UI)"
echo
echo "============================================"
echo
echo "Useful commands:"
echo "  View logs:    docker compose logs -f behav director"
echo "  Planner:      docker compose logs -f behav"
echo "  Sim:          docker compose logs -f go2_sim"
echo "  Stop:         ./stop.sh"
echo "  Stop + clean: ./stop.sh --clean"
echo
