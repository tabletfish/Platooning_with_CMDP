#!/usr/bin/env bash
set -euo pipefail
cd /home/jungjinwoo/carla_0.9.16
CARLA_RENDER_MODE="${CARLA_RENDER_MODE:-window}"
PLATOON_MAP_PATH="${PLATOON_MAP_PATH:-/Game/Carla/Maps/Town04}"
CARLA_QUALITY_LEVEL="${CARLA_QUALITY_LEVEL:-Low}"
CARLA_RES_X="${CARLA_RES_X:-960}"
CARLA_RES_Y="${CARLA_RES_Y:-540}"
CARLA_SOUND="${CARLA_SOUND:-off}"
CARLA_FPS="${CARLA_FPS:-20}"

ARGS=(--ros2 "-quality-level=${CARLA_QUALITY_LEVEL}" "-ResX=${CARLA_RES_X}" "-ResY=${CARLA_RES_Y}" "-fps=${CARLA_FPS}")
if [[ "$CARLA_RENDER_MODE" == "offscreen" ]]; then
  ARGS+=(-RenderOffScreen)
else
  ARGS+=(-windowed)
fi
if [[ "$CARLA_SOUND" == "off" ]]; then
  ARGS+=(-nosound)
fi
if [[ -n "${CARLA_EXTRA_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  EXTRA_ARGS=(${CARLA_EXTRA_ARGS})
  ARGS+=("${EXTRA_ARGS[@]}")
fi
./CarlaUE4.sh "$PLATOON_MAP_PATH" "${ARGS[@]}" "$@"
