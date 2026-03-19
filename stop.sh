#!/usr/bin/env bash
# ============================================
#  BehAV - Stop Services
# ============================================
set -euo pipefail

cd "$(dirname "$0")"

case "${1:-}" in
  --clean|-c)
    echo
    echo "============================================"
    echo "  BehAV - Stop + Full Clean"
    echo "============================================"
    echo
    echo "This will remove ALL containers, volumes and data."
    echo "(Shared bridge volume, any cached state, etc.)"
    echo
    read -rp "Are you sure? (y/N): " CONFIRM
    if [[ "${CONFIRM,,}" == "y" ]]; then
      echo
      echo "Stopping and removing everything..."
      docker-compose down -v --remove-orphans
      echo
      echo "============================================"
      echo "  Everything Removed!"
      echo "============================================"
      echo
      echo "All containers and volumes deleted."
      echo "Run ./start.sh to begin fresh."
      echo
    else
      echo
      echo "Cancelled. Nothing was removed."
      echo
    fi
    ;;
  *)
    echo
    echo "============================================"
    echo "  BehAV - Stopping Services"
    echo "============================================"
    echo
    echo "Stopping all BehAV services..."
    docker-compose stop viewer director behav bridge_reader go2_sim
    echo
    echo "============================================"
    echo "  Services Stopped!"
    echo "============================================"
    echo
    echo "Data preserved. To start again: ./start.sh"
    echo "To remove everything: ./stop.sh --clean"
    echo
    ;;
esac
