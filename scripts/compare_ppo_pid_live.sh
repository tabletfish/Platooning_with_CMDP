#!/usr/bin/env bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP
: "${PLATOON_USE_ROS:=1}"
: "${PLATOON_EVAL_EPISODES:=3}"
: "${PLATOON_EVAL_MAX_STEPS:=20}"
export PLATOON_USE_ROS PLATOON_EVAL_EPISODES PLATOON_EVAL_MAX_STEPS

echo "===== PPO ====="
PLATOON_EVAL_POLICY=ppo python3 evaluate.py

echo
echo "===== PID ====="
PLATOON_EVAL_POLICY=pid python3 evaluate.py
