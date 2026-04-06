#!/usr/bin/env bash
set -euo pipefail
cd /home/jungjinwoo/carla_0.9.16
CARLA_RENDER_MODE="${CARLA_RENDER_MODE:-offscreen}"
PLATOON_MAP_PATH="${PLATOON_MAP_PATH:-/Game/Carla/Maps/Town04}"
ARGS=(--ros2)
if [[ "$CARLA_RENDER_MODE" == "offscreen" ]]; then
  ARGS+=(-RenderOffScreen)
fi
./CarlaUE4.sh "$PLATOON_MAP_PATH" "${ARGS[@]}" "$@"
