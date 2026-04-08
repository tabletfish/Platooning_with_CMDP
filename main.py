from __future__ import annotations

import os

import omnisafe

import platoon_env  # noqa: F401  # Registers PlatoonSafe-v0 via @env_register.


def _env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def _env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def main() -> None:
    steps_per_epoch = _env_int("PLATOON_STEPS_PER_EPOCH", 2000)
    algo = os.getenv("PLATOON_ALGO", "PPOLag")

    custom_cfgs: dict = {
        "train_cfgs": {
            "total_steps": _env_int("PLATOON_TOTAL_STEPS", 1000000),
            "vector_env_nums": 1,
            "torch_threads": _env_int("PLATOON_TORCH_THREADS", 4),
        },
        "algo_cfgs": {
            "steps_per_epoch": steps_per_epoch,
            "update_iters": _env_int("PLATOON_UPDATE_ITERS", 10),
        },
        "logger_cfgs": {
            "use_wandb": False,
        },
    }
    if "lag" in algo.lower():
        custom_cfgs["lagrange_cfgs"] = {
            "cost_limit": _env_float("PLATOON_COST_LIMIT", 5.0),
        }

    print("======================================================")
    print(f"PlatoonSafe-v0 {algo} training")
    print("ROS mode:", os.getenv("PLATOON_USE_ROS", "0"))
    print("Total steps:", custom_cfgs["train_cfgs"]["total_steps"])
    if "lagrange_cfgs" in custom_cfgs:
        print("Cost limit:", custom_cfgs["lagrange_cfgs"]["cost_limit"])
    print("======================================================")

    agent = omnisafe.Agent(
        algo,
        "PlatoonSafe-v0",
        custom_cfgs=custom_cfgs,
    )
    agent.learn()


if __name__ == "__main__":
    main()
