#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" carla_bridge.py
