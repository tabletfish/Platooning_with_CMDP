# Repository Guidelines

## Project Overview

This repository trains and evaluates a safe reinforcement-learning longitudinal controller for a CAV platoon using OmniSafe, CARLA 0.9.16, and ROS2 Humble.

Key entry points:

- `main.py`: OmniSafe `PPOLag` training entry point.
- `platoon_env.py`: `PlatoonSafe-v0` CMDP environment registration and reward/cost logic.
- `ros2_node.py`: ROS2 interface for live CARLA observations and control publishing.
- `carla_bridge.py`: CARLA bridge process.
- `evaluate.py`: evaluation entry point.
- `scripts/`: smoke tests and live-run orchestration.

## Environment

- Use Python 3.10 unless a task explicitly requires otherwise.
- Python dependencies are listed in `requirements.txt`.
- Live CARLA/ROS runs expect ROS2 Humble at `/opt/ros/humble/setup.bash`.
- CARLA-dependent code targets CARLA `0.9.16`.

Do not install or upgrade dependencies unless the user asks for that or a failing verification step requires it.

## Common Commands

Run from the repository root:

```bash
cd /home/jungjinwoo/Platooning_with_CMDP
```

Mock smoke test:

```bash
./scripts/smoke_mock_train.sh
```

Live observation smoke test, after CARLA/ROS are available:

```bash
./scripts/smoke_live_step.sh
```

Live short training smoke test, after CARLA/ROS are available:

```bash
./scripts/smoke_live_train.sh
```

Full live run sequence:

```bash
./scripts/start_carla_ros2.sh
./scripts/start_bridge.sh
./scripts/train_live_full.sh
```

## Runtime Configuration

The project is configured primarily through environment variables:

- `PLATOON_USE_ROS`: set to `1` for live ROS/CARLA mode, `0` for mock mode.
- `PLATOON_TOTAL_STEPS`: total training steps.
- `PLATOON_STEPS_PER_EPOCH`: rollout steps per epoch.
- `PLATOON_UPDATE_ITERS`: OmniSafe update iterations.
- `PLATOON_MAX_EPISODE_STEPS`: episode length.
- `PLATOON_TORCH_THREADS`: Torch thread count.
- `PLATOON_COST_LIMIT`: PPOLag cost limit.

Keep new runtime knobs consistent with the existing `PLATOON_*` environment-variable pattern.

## Development Notes

- Prefer focused changes in the existing Python modules and scripts over broad refactors.
- Preserve the `PlatoonSafe-v0` environment registration behavior in `platoon_env.py`.
- Keep ROS/CARLA-specific imports inside live-mode paths when practical, so mock-mode tests remain runnable without a live simulator.
- Use Torch tensors for OmniSafe environment return values, matching the current `CMDP` implementation.
- Avoid committing generated artifacts such as `__pycache__/` and `runs/`.
- The worktree may contain local experiment changes; do not revert unrelated edits.

## Verification Guidance

- For changes that should work without CARLA/ROS, run `./scripts/smoke_mock_train.sh`.
- For ROS/CARLA integration changes, verify with `./scripts/smoke_live_step.sh` and, when practical, `./scripts/smoke_live_train.sh`.
- If live verification is not possible because CARLA or ROS2 is unavailable, state that explicitly in the final response.
