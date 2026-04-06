#!/usr/bin/env bash
set -euo pipefail
cd /home/jungjinwoo/Platooning_with_CMDP
PLATOON_TOTAL_STEPS=1000 \
PLATOON_STEPS_PER_EPOCH=500 \
PLATOON_UPDATE_ITERS=1 \
PLATOON_TORCH_THREADS=1 \
PLATOON_USE_ROS=0 \
PLATOON_MAX_EPISODE_STEPS=50 \
python3 main.py
