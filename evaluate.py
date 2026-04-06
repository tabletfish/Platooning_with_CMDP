from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import torch
from omnisafe.evaluator import Evaluator

import platoon_env  # noqa: F401
from pid_controller import LongitudinalPIDController
from platoon_env import PlatoonSafeEnv


def _latest_run_dir(base_dir: Path) -> Path:
    candidates = [path for path in base_dir.iterdir() if path.is_dir() and path.name.startswith("seed-")]
    if not candidates:
        raise FileNotFoundError(f"No run directories found under {base_dir}")
    return max(candidates, key=lambda path: path.stat().st_mtime)


def _latest_model_name(run_dir: Path) -> str:
    torch_dir = run_dir / "torch_save"
    candidates = sorted(torch_dir.glob("epoch-*.pt"))
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found under {torch_dir}")
    return candidates[-1].name


def _compute_thw(env: PlatoonSafeEnv, obs: torch.Tensor) -> float:
    delta_d = float(obs[0].item())
    ego_vel = float(obs[3].item())
    desired_distance = ego_vel * env.h + env.d_0
    actual_distance = delta_d + desired_distance
    if ego_vel <= 0.1:
        return float("inf")
    return actual_distance / ego_vel


def _build_actor(save_dir: Path | None, model_name: str | None):
    base_dir = save_dir if save_dir is not None else Path("./runs/PPOLag-{PlatoonSafe-v0}")
    run_dir = base_dir if (base_dir / "torch_save").exists() else _latest_run_dir(base_dir)
    resolved_model = model_name or _latest_model_name(run_dir)
    evaluator = Evaluator(render_mode="rgb_array")
    evaluator.load_saved(save_dir=str(run_dir), model_name=resolved_model)
    return evaluator._actor, run_dir, resolved_model


def evaluate_saved(save_dir: str | None = None, model_name: str | None = None) -> dict[str, float]:
    policy = os.getenv("PLATOON_EVAL_POLICY", "ppo").lower()
    num_episodes = int(os.getenv("PLATOON_EVAL_EPISODES", "2"))
    max_steps = int(os.getenv("PLATOON_EVAL_MAX_STEPS", os.getenv("PLATOON_MAX_EPISODE_STEPS", "1000")))

    actor = None
    run_dir = None
    resolved_model = None
    if policy == "ppo":
        actor, run_dir, resolved_model = _build_actor(Path(save_dir) if save_dir else None, model_name)
    elif policy != "pid":
        raise ValueError(f"Unsupported PLATOON_EVAL_POLICY={policy}")

    print("======================================================")
    print("PlatoonSafe-v0 custom evaluation")
    print("Policy:", policy)
    if run_dir is not None:
        print("Run dir:", run_dir)
        print("Model:", resolved_model)
    print("ROS mode:", os.getenv("PLATOON_USE_ROS", "0"))
    print("Episodes:", num_episodes)
    print("Max steps:", max_steps)
    print("======================================================")

    aggregate = {
        "episode_return": [],
        "episode_cost": [],
        "episode_length": [],
        "control_efficiency": [],
        "traffic_disturbance": [],
        "jerk_cost": [],
        "thw_cost": [],
        "caution_duration": [],
        "danger_duration": [],
    }

    for episode in range(num_episodes):
        env = PlatoonSafeEnv("PlatoonSafe-v0")
        controller = LongitudinalPIDController(dt=env.dt)
        obs, _ = env.reset(seed=episode)
        prev_action = 0.0

        episode_return = 0.0
        episode_cost = 0.0
        control_efficiency = 0.0
        traffic_disturbance = 0.0
        jerk_cost = 0.0
        thw_cost = 0.0
        caution_duration = 0.0
        danger_duration = 0.0
        episode_length = 0

        for _ in range(max_steps):
            if policy == "ppo":
                with torch.no_grad():
                    action_tensor = actor.predict(obs, deterministic=True)
                action = float(torch.as_tensor(action_tensor).reshape(-1)[0].item())
            else:
                throttle, brake = controller.compute_control(
                    spacing_error=float(obs[0].item()),
                    rel_vel=float(obs[1].item()),
                )
                action = float(throttle - brake)

            next_obs, reward, cost, terminated, truncated, _ = env.step(
                torch.tensor([action], dtype=torch.float32),
            )

            delta_d = float(next_obs[0].item())
            delta_v = float(next_obs[1].item())
            state_vec = np.array([delta_d, delta_v], dtype=np.float32)
            control_efficiency += float(state_vec.T @ env.Q @ state_vec)
            traffic_disturbance += action**2
            jerk_cost += (action - prev_action) ** 2
            thw_cost += float(cost.item())

            thw = _compute_thw(env, next_obs)
            if env.tau_danger < thw < env.tau_safe:
                caution_duration += 1.0
            elif thw <= env.tau_danger:
                danger_duration += 1.0

            episode_return += float(reward.item())
            episode_cost += float(cost.item())
            episode_length += 1
            prev_action = action
            obs = next_obs

            if bool(terminated.item()) or bool(truncated.item()):
                break

        env.close()

        aggregate["episode_return"].append(episode_return)
        aggregate["episode_cost"].append(episode_cost)
        aggregate["episode_length"].append(float(episode_length))
        aggregate["control_efficiency"].append(control_efficiency / max(episode_length, 1))
        aggregate["traffic_disturbance"].append(traffic_disturbance / max(episode_length, 1))
        aggregate["jerk_cost"].append(jerk_cost / max(episode_length, 1))
        aggregate["thw_cost"].append(thw_cost / max(episode_length, 1))
        aggregate["caution_duration"].append(caution_duration)
        aggregate["danger_duration"].append(danger_duration)

        print(f"Episode {episode + 1}: ret={episode_return:.3f}, cost={episode_cost:.3f}, len={episode_length}")

    summary = {name: float(np.mean(values)) for name, values in aggregate.items()}
    print("Average metrics:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    return summary


if __name__ == "__main__":
    evaluate_saved()
