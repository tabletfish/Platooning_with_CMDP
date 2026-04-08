# ScenarioRunner Integration

Use ScenarioRunner for the CARLA traffic scenario and keep the RL environment responsible for observations, reward, cost, and communication dropout.

## Roles

- ScenarioRunner / OpenSCENARIO: spawn the platoon, drive the leader profile, schedule repeated braking events.
- `carla_bridge.py`: attach to ScenarioRunner vehicles, apply RL control to followers, publish follower state.
- `platoon_env.py`: compute observation, reward, THW cost, termination, and logging.

## Run

Install or clone CARLA ScenarioRunner so that `scenario_runner.py` exists. This workstation uses:

```bash
SCENARIO_RUNNER_PATH=/home/jungjinwoo/scenario_runner
```

Then run:

```bash
./scripts/run_scenario_runner_live.sh
```

The script generates:

```text
scenarios/generated/platoon_longitudinal_brake.xosc
```

It also sets:

```bash
PLATOON_USE_SCENARIO_RUNNER=1
PLATOON_RESET_CARLA_ON_ENV_RESET=0
```

Those settings prevent the bridge from fighting ScenarioRunner over leader control and respawn.

## Baselines

Use OmniSafe for constrained and unconstrained RL:

```bash
./scripts/train_live_full.sh
./scripts/train_ppo_live_full.sh
```

Use PID through `evaluate.py`:

```bash
PLATOON_EVAL_POLICY=pid PLATOON_EVAL_TRACE_CSV=logs/pid_trace.csv python3.10 evaluate.py
```

Stable-Baselines3 is listed as an optional dependency, but it should not be used directly on `PlatoonSafeEnv` without a Gymnasium wrapper because the current environment is an OmniSafe CMDP with separate reward and cost.

This repo includes `sb3_env.py` for an unconstrained SB3 PPO baseline:

```bash
./scripts/train_sb3_ppo.sh
PLATOON_EVAL_POLICY=sb3 PLATOON_SB3_MODEL=runs/SB3-PPO-{PlatoonSafe-v0}/ppo_platoon.zip python3.10 evaluate.py
```

Interpret SB3 PPO as a reward-only baseline. It logs cost for evaluation but does not optimize a Lagrangian constraint.
