#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
: "${PLATOON_USE_ROS:=1}"
: "${PLATOON_TOTAL_STEPS:=1000000}"
: "${PLATOON_MAX_EPISODE_STEPS:=2000}"
export PLATOON_USE_ROS PLATOON_TOTAL_STEPS PLATOON_MAX_EPISODE_STEPS
"$PYTHON_BIN" train_sb3.py
