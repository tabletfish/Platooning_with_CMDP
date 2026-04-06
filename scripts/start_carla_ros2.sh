#!/usr/bin/env bash
set -euo pipefail
cd /home/jungjinwoo/carla_0.9.16
./CarlaUE4.sh --ros2 -RenderOffScreen "$@"
