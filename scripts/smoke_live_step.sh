#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP
PYTHON_BIN="${PYTHON_BIN:-python3.10}"
PLATOON_USE_ROS=1 \
PLATOON_MAX_EPISODE_STEPS=2000 \
PLATOON_RESET_CARLA_ON_ENV_RESET=0 \
"$PYTHON_BIN" - <<"PY"
import time
import torch
from platoon_env import PlatoonSafeEnv

env = PlatoonSafeEnv("PlatoonSafe-v0")
obs, info = env.reset(seed=1)
print("reset", obs.tolist())
for i in range(5):
    step_out = env.step(torch.tensor([0.05]))
    print("step", i, step_out[0].tolist(), float(step_out[1]), float(step_out[2]))
    time.sleep(0.1)
env.close()
PY
