#!/usr/bin/env bash
set -eo pipefail
set +u
source /opt/ros/humble/setup.bash
set -u
cd /home/jungjinwoo/Platooning_with_CMDP

: "${PLATOON_USE_ROS:=1}"
: "${PLATOON_EXPERIMENT_PRESET:=}"
if [[ -n "$PLATOON_EXPERIMENT_PRESET" ]]; then
  PRESET_FILE="scripts/presets/${PLATOON_EXPERIMENT_PRESET}.env"
  if [[ ! -f "$PRESET_FILE" ]]; then
    echo "Preset not found: $PRESET_FILE" >&2
    exit 1
  fi
  set -a
  source "$PRESET_FILE"
  set +a
fi

: "${PLATOON_TOTAL_STEPS:=1000}"
: "${PLATOON_STEPS_PER_EPOCH:=500}"
: "${PLATOON_UPDATE_ITERS:=1}"
: "${PLATOON_TORCH_THREADS:=1}"
: "${PLATOON_MAX_EPISODE_STEPS:=50}"
: "${PLATOON_EVAL_EPISODES:=2}"
: "${PLATOON_EVAL_MAX_STEPS:=${PLATOON_MAX_EPISODE_STEPS}}"
export PLATOON_USE_ROS PLATOON_TOTAL_STEPS PLATOON_STEPS_PER_EPOCH PLATOON_UPDATE_ITERS PLATOON_TORCH_THREADS PLATOON_MAX_EPISODE_STEPS PLATOON_EVAL_EPISODES PLATOON_EVAL_MAX_STEPS

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
"$PYTHON_BIN" - <<"PY"
import carla
client = carla.Client("localhost", 2000)
client.set_timeout(2.0)
world = client.get_world()
print("CARLA_OK", world.get_map().name)
PY

RUN_ID=$(date +%Y%m%d_%H%M%S)
LOG_DIR="logs/exp_${RUN_ID}"
mkdir -p "$LOG_DIR"
BRIDGE_LOG="$LOG_DIR/bridge.log"
TRAIN_LOG="$LOG_DIR/train.log"
PPO_EVAL_LOG="$LOG_DIR/ppo_eval.log"
PID_EVAL_LOG="$LOG_DIR/pid_eval.log"
SUMMARY_LOG="$LOG_DIR/summary.txt"

"$PYTHON_BIN" carla_bridge.py > "$BRIDGE_LOG" 2>&1 &
BRIDGE_PID=$!
cleanup() {
  kill "$BRIDGE_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT
sleep 3

echo "Preset: ${PLATOON_EXPERIMENT_PRESET:-manual}"
echo "===== TRAIN PPO ====="
"$PYTHON_BIN" main.py | tee "$TRAIN_LOG"

echo
echo "===== EVAL PPO ====="
PLATOON_EVAL_POLICY=ppo "$PYTHON_BIN" evaluate.py | tee "$PPO_EVAL_LOG"

echo
echo "===== EVAL PID ====="
PLATOON_EVAL_POLICY=pid "$PYTHON_BIN" evaluate.py | tee "$PID_EVAL_LOG"

{
  echo "Run ID: $RUN_ID"
  echo "Preset: ${PLATOON_EXPERIMENT_PRESET:-manual}"
  echo "Bridge log: $BRIDGE_LOG"
  echo "Train log: $TRAIN_LOG"
  echo "PPO eval log: $PPO_EVAL_LOG"
  echo "PID eval log: $PID_EVAL_LOG"
  echo
  echo "--- PPO summary ---"
  grep -E "episode_return:|episode_cost:|control_efficiency:|traffic_disturbance:|jerk_cost:|thw_cost:|danger_duration:" "$PPO_EVAL_LOG" || true
  echo
  echo "--- PID summary ---"
  grep -E "episode_return:|episode_cost:|control_efficiency:|traffic_disturbance:|jerk_cost:|thw_cost:|danger_duration:" "$PID_EVAL_LOG" || true
} | tee "$SUMMARY_LOG"

echo
echo "Summary log: $SUMMARY_LOG"
