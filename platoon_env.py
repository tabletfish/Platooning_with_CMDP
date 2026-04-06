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
    """Custom OmniSafe CMDP for longitudinal platoon control."""

    _support_envs: ClassVar[list[str]] = ["PlatoonSafe-v0"]
    _action_space: OmnisafeSpace
    _observation_space: OmnisafeSpace
    _metadata: ClassVar[dict[str, Any]] = {}

    need_auto_reset_wrapper = True
    need_time_limit_wrapper = True
    _num_envs = 1

    def __init__(self, env_id: str, **kwargs: Any) -> None:
        super().__init__(env_id)
        del kwargs

        self._action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self._observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        self._max_episode_steps = int(os.getenv("PLATOON_MAX_EPISODE_STEPS", "3000"))

        self.dt = float(os.getenv("PLATOON_DT", "0.05"))
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
        self.collision_distance = float(os.getenv("PLATOON_COLLISION_DISTANCE", "1.5"))
        self.brake_gain = float(os.getenv("PLATOON_MOCK_BRAKE_GAIN", "4.0"))
        self.drag_gain = float(os.getenv("PLATOON_MOCK_DRAG_GAIN", "0.08"))
        self.mock_leader_accel_gain = float(os.getenv("PLATOON_MOCK_LEADER_ACCEL_GAIN", "0.6"))
        self.mock_lat_damping = float(os.getenv("PLATOON_MOCK_LAT_DAMPING", "1.5"))
        self._leader_speed_candidates = (15.0, 17.5, 20.0)

        self.spacing_error_clip = float(os.getenv("PLATOON_SPACING_ERROR_CLIP", "20.0"))
        self.rel_vel_clip = float(os.getenv("PLATOON_REL_VEL_CLIP", "10.0"))
        self.accel_clip = float(os.getenv("PLATOON_ACCEL_CLIP", "5.0"))
        self.ego_vel_clip = float(os.getenv("PLATOON_EGO_VEL_CLIP", "40.0"))
        self.lat_offset_clip = float(os.getenv("PLATOON_LAT_OFFSET_CLIP", "5.0"))

        self.use_ros = os.getenv("PLATOON_USE_ROS", "0") == "1"
        self._ros_node = None
        self._owns_rclpy = False
        self._rng = random.Random()
        self._leader_profile = self._sample_leader_profile()

        self.last_comm_data = self._initial_comm_state()
        self._mock_state = self._initial_mock_state()
        self.time_since_last_comm = 0.0
        self.prev_accel = 0.0
        self._count = 0
        self.env_spec_log = {"Env/CommFailures": 0, "Env/UnsafeSteps": 0}

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

        ros_received = False
        if self._ros_node is not None:
            self._ros_node.publish_control(throttle, brake)
            ros_received = self._ros_node.tick_and_wait(self.dt)

        simulated_dropout = self._rng.random() > self.p_success
        obs_np = self._get_observation(
            action_value=action_value,
            ros_received=ros_received,
            simulated_dropout=simulated_dropout,
        )
        effective_comm_success = ros_received and not simulated_dropout if self._ros_node is not None else not simulated_dropout
        if not effective_comm_success:
            self.env_spec_log["Env/CommFailures"] += 1

        delta_d = float(obs_np[0])
        delta_v = float(obs_np[1])
        ego_vel = float(obs_np[3])

        reward = self._compute_reward(delta_d, delta_v, action_value, self.prev_accel)
        cost = self._compute_cost(delta_d, ego_vel)
        self.prev_accel = action_value

        actual_distance = delta_d + ego_vel * self.h + self.d_0
        terminated = actual_distance <= self.collision_distance
        truncated = self._count >= self._max_episode_steps
        if terminated:
            self.env_spec_log["Env/UnsafeSteps"] += 1
        obs = torch.as_tensor(obs_np, dtype=torch.float32)
        info = {
            "cost": cost,
            "comm_success": effective_comm_success,
            "transport_success": ros_received if self._ros_node is not None else True,
            "simulated_dropout": simulated_dropout,
            "throttle": throttle,
            "brake": brake,
            "time_since_last_comm": self.time_since_last_comm,
            "actual_distance": actual_distance,
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
        self.last_comm_data = self._initial_comm_state()
        self._mock_state = self._initial_mock_state()
        self._leader_profile = self._sample_leader_profile()
        if self._ros_node is not None:
            self._ros_node.publish_reset()
            self._ros_node.tick_and_wait(self.dt)
        obs = torch.as_tensor(
            self._get_observation(
                action_value=0.0,
                ros_received=self._ros_node is None or self._ros_node.get_latest_data_if_fresh(self.dt * 2.0) is not None,
                simulated_dropout=False,
            ),
            dtype=torch.float32,
        )
        return obs, {}

    @property
    def max_episode_steps(self) -> int:
        return self._max_episode_steps

    def spec_log(self, logger: Logger) -> None:
        logger.store(self.env_spec_log)
        self.env_spec_log = {"Env/CommFailures": 0, "Env/UnsafeSteps": 0}

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        self._rng.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _initial_comm_state(self) -> dict[str, float]:
        return {
            "spacing_error": 2.0,
            "rel_vel": 0.0,
            "prec_accel": 0.0,
            "ego_vel": 12.0,
            "lat_offset": 0.0,
        }

    def _initial_mock_state(self) -> dict[str, float]:
        return {
            "actual_distance": self.d_0 + self.h * 12.0 + 2.0,
            "ego_vel": 0.0,
            "leader_vel": 0.0,
            "leader_accel": 0.0,
            "lat_offset": 0.0,
            "leader_profile_step": 0.0,
        }

    def _sample_leader_profile(self) -> dict[str, float]:
        target_speed = self._leader_speed_candidates[self._rng.randrange(len(self._leader_speed_candidates))]
        return {
            "target_speed": target_speed,
            "cruise_steps": float(int(os.getenv("PLATOON_CRUISE_STEPS", "120"))),
            "brake_steps": float(int(os.getenv("PLATOON_BRAKE_STEPS", "40"))),
        }

    def _clip_state(self, key: str, value: float) -> float:
        if not np.isfinite(value):
            value = self.last_comm_data.get(key, 0.0)
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

    def _simulate_mock_step(self, action_value: float) -> None:
        ego_accel = float(np.clip(action_value * self.accel_clip, -self.brake_gain, self.accel_clip))
        profile_step = int(self._mock_state["leader_profile_step"])
        target_speed = float(self._leader_profile["target_speed"])
        cruise_steps = int(self._leader_profile["cruise_steps"])
        brake_steps = int(self._leader_profile["brake_steps"])
        leader_vel = float(self._mock_state["leader_vel"])
        ego_vel = float(self._mock_state["ego_vel"])

        if leader_vel < target_speed and profile_step < 200:
            leader_accel = self.mock_leader_accel_gain * (target_speed - leader_vel)
        elif profile_step < 200 + cruise_steps:
            leader_accel = 0.0
        elif profile_step < 200 + cruise_steps + brake_steps:
            leader_accel = -4.0
        else:
            leader_accel = 0.0
        leader_vel = float(np.clip(leader_vel + leader_accel * self.dt, 0.0, self.ego_vel_clip))
        ego_vel = float(
            np.clip(
                ego_vel + (ego_accel - self.drag_gain * ego_vel) * self.dt,
                0.0,
                self.ego_vel_clip,
            ),
        )
        actual_distance = float(
            np.clip(
                self._mock_state["actual_distance"] + (leader_vel - ego_vel) * self.dt,
                0.0,
                200.0,
            ),
        )
        lat_offset = float(
            np.clip(
                self._mock_state["lat_offset"] * (1.0 - self.mock_lat_damping * self.dt)
                + 0.02 * self._rng.uniform(-1.0, 1.0),
                -self.lat_offset_clip,
                self.lat_offset_clip,
            ),
        )
        desired_distance = ego_vel * self.h + self.d_0
        self._mock_state.update(
            {
                "actual_distance": actual_distance,
                "ego_vel": ego_vel,
                "leader_vel": leader_vel,
                "leader_accel": leader_accel,
                "lat_offset": lat_offset,
                "leader_profile_step": profile_step + 1,
            },
        )
        self.last_comm_data = {
            "spacing_error": self._clip_state("spacing_error", actual_distance - desired_distance),
            "rel_vel": self._clip_state("rel_vel", leader_vel - ego_vel),
            "prec_accel": self._clip_state("prec_accel", leader_accel),
            "ego_vel": self._clip_state("ego_vel", ego_vel),
            "lat_offset": self._clip_state("lat_offset", lat_offset),
        }

    def _get_observation(self, action_value: float, ros_received: bool, simulated_dropout: bool) -> np.ndarray:
        if self._ros_node is None:
            self._simulate_mock_step(action_value)
            ros_received = True

        fresh_data = None
        if self._ros_node is not None and ros_received and not simulated_dropout:
            fresh_data = self._ros_node.get_latest_data_if_fresh(self.dt * 2.0)

        if fresh_data is not None:
            for key, value in fresh_data.items():
                self.last_comm_data[key] = self._clip_state(key, value)
            self.time_since_last_comm = 0.0
        elif self._ros_node is not None:
            self.time_since_last_comm += self.dt

        effective_failure = simulated_dropout or fresh_data is None
        if self._ros_node is None:
            effective_failure = simulated_dropout
            if not simulated_dropout:
                self.time_since_last_comm = 0.0
            else:
                self.time_since_last_comm += self.dt

        failure_flag = 1.0 if effective_failure else 0.0
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
        return np.nan_to_num(obs, nan=0.0, posinf=self.spacing_error_clip, neginf=-self.spacing_error_clip)

    def _compute_reward(self, delta_d: float, delta_v: float, action: float, prev_action: float) -> float:
        delta_d = 0.0 if not np.isfinite(delta_d) else delta_d
        delta_v = 0.0 if not np.isfinite(delta_v) else delta_v
        action = 0.0 if not np.isfinite(action) else action
        prev_action = 0.0 if not np.isfinite(prev_action) else prev_action
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
        delta_d = 0.0 if not np.isfinite(delta_d) else delta_d
        ego_vel = 0.0 if not np.isfinite(ego_vel) else ego_vel
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
