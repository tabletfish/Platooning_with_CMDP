from __future__ import annotations

import os
from pathlib import Path

from sb3_env import PlatoonGymEnv


def main() -> None:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3.common.monitor import Monitor
    except ImportError as exc:
        raise SystemExit(
            "stable-baselines3 is not installed. Install requirements or run: "
            "python3.10 -m pip install stable-baselines3",
        ) from exc

    total_steps = int(os.getenv("PLATOON_SB3_TOTAL_STEPS", os.getenv("PLATOON_TOTAL_STEPS", "1000000")))
    log_dir = Path(os.getenv("PLATOON_SB3_LOG_DIR", "runs/SB3-PPO-{PlatoonSafe-v0}"))
    model_path = log_dir / "ppo_platoon.zip"
    log_dir.mkdir(parents=True, exist_ok=True)

    env = Monitor(PlatoonGymEnv(), filename=str(log_dir / "monitor.csv"))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log=str(log_dir / "tb"),
        n_steps=int(os.getenv("PLATOON_SB3_N_STEPS", "2048")),
        batch_size=int(os.getenv("PLATOON_SB3_BATCH_SIZE", "64")),
        learning_rate=float(os.getenv("PLATOON_SB3_LR", "0.0003")),
    )
    model.learn(total_timesteps=total_steps)
    model.save(model_path)
    env.close()
    print(model_path)


if __name__ == "__main__":
    main()
