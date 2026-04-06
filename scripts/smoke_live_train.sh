#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
PLATOON_TOTAL_STEPS=100 \
PLATOON_STEPS_PER_EPOCH=50 \
PLATOON_UPDATE_ITERS=1 \
PLATOON_TORCH_THREADS=1 \
PLATOON_USE_ROS=1 \
PLATOON_MAX_EPISODE_STEPS=10 \
"$PYTHON_BIN" main.py
