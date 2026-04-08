#!/usr/bin/env bash
set -euo pipefail
set +u
source /opt/ros/humble/setup.bash
if [[ -f /home/jungjinwoo/carla-ros-bridge/install/setup.bash ]]; then
  source /home/jungjinwoo/carla-ros-bridge/install/setup.bash
fi
set -u
cd /home/jungjinwoo/Platooning_with_CMDP

PYTHON_BIN="${PYTHON_BIN:-python3.10}"
: "${SCENARIO_RUNNER_PATH:=/home/jungjinwoo/scenario_runner}"
: "${PLATOON_SCENARIO_SEED:=0}"
: "${PLATOON_SCENARIO_FILE:=scenarios/generated/platoon_longitudinal_brake.xosc}"
: "${PLATOON_USE_ROS:=1}"
: "${PLATOON_USE_SCENARIO_RUNNER:=1}"
: "${PLATOON_RESET_CARLA_ON_ENV_RESET:=0}"
: "${PLATOON_TOTAL_STEPS:=1000000}"
: "${PLATOON_STEPS_PER_EPOCH:=2000}"
: "${PLATOON_UPDATE_ITERS:=10}"
: "${PLATOON_TORCH_THREADS:=4}"
: "${PLATOON_MAX_EPISODE_STEPS:=2000}"
: "${PLATOON_HARD_VIOLATION_DISTANCE:=500.0}"
: "${PLATOON_SPECTATOR_FOLLOW:=1}"
: "${PLATOON_STOP_HOLD_SECONDS_MIN:=3.0}"
: "${PLATOON_STOP_HOLD_SECONDS_MAX:=7.0}"
export PLATOON_USE_ROS PLATOON_USE_SCENARIO_RUNNER PLATOON_RESET_CARLA_ON_ENV_RESET PLATOON_TOTAL_STEPS PLATOON_STEPS_PER_EPOCH PLATOON_UPDATE_ITERS PLATOON_TORCH_THREADS PLATOON_MAX_EPISODE_STEPS PLATOON_HARD_VIOLATION_DISTANCE PLATOON_SPECTATOR_FOLLOW PLATOON_STOP_HOLD_SECONDS_MIN PLATOON_STOP_HOLD_SECONDS_MAX
export PYTHONPATH="$SCENARIO_RUNNER_PATH:${PYTHONPATH:-}"

if [[ -z "$SCENARIO_RUNNER_PATH" || ! -f "$SCENARIO_RUNNER_PATH/scenario_runner.py" ]]; then
  echo "SCENARIO_RUNNER_PATH must point to a ScenarioRunner checkout containing scenario_runner.py" >&2
  echo "Example: SCENARIO_RUNNER_PATH=/home/jungjinwoo/scenario_runner ./scripts/run_scenario_runner_live.sh" >&2
  exit 2
fi

SCENARIO_PATH="$("$PYTHON_BIN" scripts/generate_platoon_xosc.py \
  --seed "$PLATOON_SCENARIO_SEED" \
  --brake-target-min 0.0 \
  --brake-target-max 0.0 \
  --stop-hold-min "$PLATOON_STOP_HOLD_SECONDS_MIN" \
  --stop-hold-max "$PLATOON_STOP_HOLD_SECONDS_MAX" \
  --output "$PLATOON_SCENARIO_FILE")"
SCENARIO_ABS="$(readlink -f "$SCENARIO_PATH")"

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/scenario_runner_${RUN_ID}"
mkdir -p "$LOG_DIR"

ros2 launch carla_ros_scenario_runner carla_ros_scenario_runner.launch.py \
  scenario_runner_path:="$SCENARIO_RUNNER_PATH" \
  role_name:=leader \
  wait_for_ego:= \
  > "$LOG_DIR/scenario_runner.log" 2>&1 &
SCENARIO_RUNNER_PID=$!

PLATOON_USE_SCENARIO_RUNNER=1 "$PYTHON_BIN" carla_bridge.py > "$LOG_DIR/bridge.log" 2>&1 &
BRIDGE_PID=$!

cleanup() {
  kill "$BRIDGE_PID" >/dev/null 2>&1 || true
  kill "$SCENARIO_RUNNER_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT

sleep 3
ros2 service call /scenario_runner/execute_scenario carla_ros_scenario_runner_types/srv/ExecuteScenario \
  "{scenario: {name: platoon_longitudinal_brake, scenario_file: '$SCENARIO_ABS'}}" \
  > "$LOG_DIR/execute_scenario.log"

echo "Scenario: $SCENARIO_ABS"
echo "Logs: $LOG_DIR"
"$PYTHON_BIN" main.py | tee "$LOG_DIR/train.log"
