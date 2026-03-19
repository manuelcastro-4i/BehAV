#!/usr/bin/env bash
# ============================================
#  BehAV - Build Services
# ============================================
set -euo pipefail

cd "$(dirname "$0")"

print_header() { echo; echo "============================================"; echo "  BehAV - $1"; echo "============================================"; echo; }
print_done()   { print_header "Build Complete!"; echo "Usage:"; echo "  ./build.sh                Build behav image (planner + director + viewer)"; echo "  ./build.sh --sim          Build go2_sim image (Gazebo + ROS2 Jazzy)"; echo "  ./build.sh --all          Build all images"; echo "  ./build.sh --no-cache     Build behav without cache (full rebuild)"; echo "  ./build.sh --all --no-cache  Build all without cache"; echo; echo "To start: ./start.sh"; echo; }

NO_CACHE=""
[[ "${2:-}" == "--no-cache" ]] && NO_CACHE="--no-cache"

case "${1:-}" in
  --no-cache|-nc)
    print_header "Building behav (no cache)..."
    docker-compose build --no-cache behav
    ;;
  --sim|-s)
    print_header "Building go2_sim..."
    docker-compose build $NO_CACHE go2_sim
    ;;
  --all|-a)
    print_header "Building ALL images..."
    docker-compose build $NO_CACHE
    ;;
  *)
    # Default: behav image (planner, director, bridge_reader, viewer all use it)
    print_header "Building behav image..."
    docker compose build behav
    ;;
esac

print_done
