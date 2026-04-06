#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
cd /home/jungjinwoo/Platooning_with_CMDP
PLATOON_TOTAL_STEPS=100 \
PLATOON_STEPS_PER_EPOCH=50 \
PLATOON_UPDATE_ITERS=1 \
PLATOON_TORCH_THREADS=1 \
PLATOON_USE_ROS=1 \
PLATOON_MAX_EPISODE_STEPS=10 \
python3 main.py
