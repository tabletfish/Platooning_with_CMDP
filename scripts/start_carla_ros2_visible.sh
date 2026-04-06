#!/usr/bin/env bash
set -euo pipefail
cd /home/jungjinwoo/Platooning_with_CMDP
CARLA_RENDER_MODE=window ./scripts/start_carla_ros2.sh "$@"
