#!/usr/bin/env bash
set -euo pipefail
source /opt/ros/humble/setup.bash
cd /home/jungjinwoo/Platooning_with_CMDP
: "${PLATOON_USE_ROS:=1}"
: "${PLATOON_MAX_EPISODE_STEPS:=2000}"
: "${PLATOON_EVAL_EPISODES:=2}"
: "${PLATOON_EVAL_MAX_STEPS:=${PLATOON_MAX_EPISODE_STEPS}}"
export PLATOON_USE_ROS PLATOON_MAX_EPISODE_STEPS PLATOON_EVAL_EPISODES PLATOON_EVAL_MAX_STEPS
python3 evaluate.py
