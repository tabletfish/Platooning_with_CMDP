from __future__ import annotations

import os
import random
from typing import Any, ClassVar

import numpy as np
import torch
from gymnasium import spaces

from omnisafe.common.logger import Logger
from omnisafe.envs.core import CMDP, env_register
from omnisafe.typing import OmnisafeSpace


@env_register
class PlatoonSafeEnv(CMDP):
    """Minimal OmniSafe-compatible platoon environment."""

    _support_envs: ClassVar[list[str]] = ["PlatoonSafe-v0"]
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    metadata: ClassVar[dict[str, int]] = {}

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        del env_id, kwargs

        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self._max_episode_steps = int(os.getenv("PLATOON_MAX_EPISODE_STEPS", "3000"))

        self.dt = 0.05
        self.p_success = float(os.getenv("PLATOON_COMM_SUCCESS", "0.9"))
        self.h = 1.0
        self.d_0 = 7.0
        self.omega_1 = 0.1
        self.omega_2 = 0.25
        self.omega_3 = 0.25
        self.Q = np.array([[2.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        self.d_far = 8.0
        self.d_close = -4.0
        self.alpha_1 = 20.0
        self.alpha_2 = 10.0
        self.tau_safe = 1.2
        self.tau_danger = 1.0

        self.spacing_error_clip = float(os.getenv("PLATOON_SPACING_ERROR_CLIP", "20.0"))
        self.rel_vel_clip = float(os.getenv("PLATOON_REL_VEL_CLIP", "10.0"))
        self.accel_clip = float(os.getenv("PLATOON_ACCEL_CLIP", "5.0"))
        self.ego_vel_clip = float(os.getenv("PLATOON_EGO_VEL_CLIP", "40.0"))
        self.lat_offset_clip = float(os.getenv("PLATOON_LAT_OFFSET_CLIP", "5.0"))

        self.use_ros = os.getenv("PLATOON_USE_ROS", "0") == "1"
        self._ros_node = None
        self._owns_rclpy = False

        self.last_comm_data = {
            "spacing_error": 0.0,
            "rel_vel": 0.0,
            "prec_accel": 0.0,
            "ego_vel": 0.0,
            "lat_offset": 0.0,
        }
        self.time_since_last_comm = 0.0
        self.prev_accel = 0.0
        self._count = 0
        self.env_spec_log = {"Env/CommFailures": 0}

        if self.use_ros:
            self._setup_ros_node()

    def _setup_ros_node(self) -> None:
        import rclpy
        from ros2_node import PlatoonROS2Node

        if not rclpy.ok():
            rclpy.init()
            self._owns_rclpy = True
        self._ros_node = PlatoonROS2Node()

    def step(
        self,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        self._count += 1

        action_value = float(torch.as_tensor(action).reshape(-1)[0].item())
        throttle = max(action_value, 0.0)
        brake = max(-action_value, 0.0)

        if self._ros_node is not None:
            self._ros_node.publish_control(throttle, brake)
            self._ros_node.tick_and_wait(self.dt)

        comm_success = random.random() <= self.p_success
        if not comm_success:
            self.env_spec_log["Env/CommFailures"] += 1

        obs_np = self._get_observation(comm_success)
        delta_d = float(obs_np[0])
        delta_v = float(obs_np[1])
        ego_vel = float(obs_np[3])

        reward = self._compute_reward(delta_d, delta_v, action_value, self.prev_accel)
        cost = self._compute_cost(delta_d, ego_vel)
        self.prev_accel = action_value

        terminated = False
        truncated = self._count >= self._max_episode_steps
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        info = {
            "cost": cost,
            "comm_success": comm_success,
            "throttle": throttle,
            "brake": brake,
        }
        return (
            obs,
            torch.as_tensor(reward, dtype=torch.float32),
            torch.as_tensor(cost, dtype=torch.float32),
            torch.as_tensor(terminated),
            torch.as_tensor(truncated),
            info,
        )

    def reset(
        self,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, dict]:
        del options
        if seed is not None:
            self.set_seed(seed)

        self._count = 0
        self.prev_accel = 0.0
        self.time_since_last_comm = 0.0
        self.last_comm_data = {
            "spacing_error": 0.0,
            "rel_vel": 0.0,
            "prec_accel": 0.0,
            "ego_vel": 0.0,
            "lat_offset": 0.0,
        }
        obs = torch.as_tensor(self._get_observation(comm_success=True), dtype=torch.float32)
        return obs, {}

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Logger) -> None:
        logger.store(self.env_spec_log)
        self.env_spec_log = {"Env/CommFailures": 0}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _clip_state(self, key: str, value: float) -> float:
        if key == "spacing_error":
            return float(np.clip(value, -self.spacing_error_clip, self.spacing_error_clip))
        if key == "rel_vel":
            return float(np.clip(value, -self.rel_vel_clip, self.rel_vel_clip))
        if key == "prec_accel":
            return float(np.clip(value, -self.accel_clip, self.accel_clip))
        if key == "ego_vel":
            return float(np.clip(value, 0.0, self.ego_vel_clip))
        if key == "lat_offset":
            return float(np.clip(value, -self.lat_offset_clip, self.lat_offset_clip))
        return float(value)

    def _get_observation(self, comm_success: bool) -> np.ndarray:
        if self._ros_node is not None and comm_success:
            latest = self._ros_node.get_latest_data()
            if latest:
                for key, value in latest.items():
                    self.last_comm_data[key] = self._clip_state(key, value)
                self.time_since_last_comm = 0.0
        elif not comm_success:
            self.time_since_last_comm += self.dt

        failure_flag = 0.0 if comm_success else 1.0
        obs = np.array(
            [
                self.last_comm_data["spacing_error"],
                self.last_comm_data["rel_vel"],
                self.last_comm_data["prec_accel"],
                self.last_comm_data["ego_vel"],
                self.last_comm_data["lat_offset"],
                failure_flag,
                float(np.clip(self.time_since_last_comm, 0.0, 5.0)),
            ],
            dtype=np.float32,
        )
        return obs

    def _compute_reward(self, delta_d: float, delta_v: float, action: float, prev_action: float) -> float:
        delta_d = float(np.clip(delta_d, -self.spacing_error_clip, self.spacing_error_clip))
        delta_v = float(np.clip(delta_v, -self.rel_vel_clip, self.rel_vel_clip))
        state_vec = np.array([delta_d, delta_v], dtype=np.float32)
        r_cont = float(state_vec.T @ self.Q @ state_vec)
        r_traf = action**2
        r_jerk = (action - prev_action) ** 2

        if delta_d > self.d_far:
            r_penalty = ((delta_d - self.d_far) ** 2) / self.alpha_1
        elif delta_d < self.d_close:
            r_penalty = ((delta_d - self.d_close) ** 2) / self.alpha_2
        else:
            r_penalty = 0.0

        reward = np.exp(-(self.omega_1 * r_cont + self.omega_2 * r_traf + self.omega_3 * r_jerk)) - r_penalty
        return float(reward)

    def _compute_cost(self, delta_d: float, ego_vel: float) -> float:
        d_ctg = ego_vel * self.h + self.d_0
        actual_distance = delta_d + d_ctg
        thw = actual_distance / ego_vel if ego_vel > 0.1 else float("inf")

        if thw >= self.tau_safe:
            cost = 0.0
        elif thw <= self.tau_danger:
            cost = 1.0
        else:
            ratio = (self.tau_safe - thw) / (self.tau_safe - self.tau_danger)
            cost = ratio**2
        return float(cost)

    def render(self) -> Any:
        return None

    def close(self) -> None:
        if self._ros_node is not None:
            self._ros_node.destroy_node()
            self._ros_node = None
        if self._owns_rclpy:
            import rclpy
            if rclpy.ok():
                rclpy.shutdown()
            self._owns_rclpy = False
