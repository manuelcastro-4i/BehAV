#!/usr/bin/env bash
# ============================================
#  BehAV - Start Services
# ============================================
set -euo pipefail

cd "$(dirname "$0")"

echo
echo "============================================"
echo "  BehAV - Starting Services"
echo "============================================"
echo

# Remove stale go2_sim / bridge_reader containers so Xvfb doesn't hit a lock
# file left over from a previous run. Other containers are recreated only if
# their definition changed.
echo "Removing stale sim containers (prevents Xvfb /tmp/.X99-lock issues)..."
docker-compose rm -f go2_sim bridge_reader 2>/dev/null || true

# Clear stale cmd_vel.json so the robot doesn't shoot off before Gazebo settles
echo "Clearing shared volume..."
docker run --rm -v behav_shared_bridge:/shared alpine \
  sh -c "rm -f /shared/cmd_vel.json" 2>/dev/null || true

echo
echo "Starting all BehAV services..."
docker-compose up -d go2_sim bridge_reader behav director viewer

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
echo "  Gazebo GUI (noVNC):    http://localhost:6080"
echo "  Costmap Viewer:        http://localhost:8080"
echo
echo "  Instruction:           ${BEHAV_INSTRUCTION:-Stay on the pavement, avoid the grass, and go to the red marker at 17 meters}"
echo "  Goal radius:           ${BEHAV_GOAL_RADIUS:-17.0} m"
echo
echo "============================================"
echo
echo "Useful commands:"
echo "  View logs:    docker-compose logs -f behav director"
echo "  Planner:      docker-compose logs -f behav"
echo "  Sim:          docker-compose logs -f go2_sim"
echo "  Stop:         ./stop.sh"
echo "  Stop + clean: ./stop.sh --clean"
echo
