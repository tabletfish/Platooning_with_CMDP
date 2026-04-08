from __future__ import annotations

from typing import Any

import gymnasium as gym
import numpy as np
import torch

from platoon_env import PlatoonSafeEnv


class PlatoonGymEnv(gym.Env):
    """Gymnasium wrapper for unconstrained SB3 baselines."""

    metadata = {"render_modes": []}

    def __init__(self, env_id: str = "PlatoonSafe-v0") -> None:
        super().__init__()
        self.env = PlatoonSafeEnv(env_id)
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        return np.asarray(obs, dtype=np.float32), info

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        obs, reward, cost, terminated, truncated, info = self.env.step(torch.as_tensor(action, dtype=torch.float32))
        info = dict(info)
        info["cost"] = float(cost.item())
        return (
            np.asarray(obs, dtype=np.float32),
            float(reward.item()),
            bool(terminated.item()),
            bool(truncated.item()),
            info,
        )

    def render(self) -> None:
        return None

    def close(self) -> None:
        self.env.close()
