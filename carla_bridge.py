from __future__ import annotations

import json
import math
import os
import sys
import time
from typing import Iterable

import carla
import numpy as np
import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy._rclpy_pybind11 import RCLError
from std_msgs.msg import Empty, Float32MultiArray

CARLA_AGENT_PATH = os.path.join(os.path.dirname(__file__), "..", "carla_0.9.16", "PythonAPI", "carla")
if CARLA_AGENT_PATH not in sys.path:
    sys.path.insert(0, CARLA_AGENT_PATH)

from agents.navigation.controller import PIDLateralController


def _velocity_norm(vector: carla.Vector3D) -> float:
    return float(math.sqrt(vector.x**2 + vector.y**2 + vector.z**2))


class CarlaPlatoonBridge(Node):
    def __init__(self) -> None:
        super().__init__("carla_platoon_bridge")
        self._shutdown_requested = False
        self._stack_config = self._load_stack_config()

        self.client = carla.Client(os.getenv("CARLA_HOST", "localhost"), int(os.getenv("CARLA_PORT", "2000")))
        self.client.set_timeout(float(os.getenv("CARLA_TIMEOUT", "10.0")))

        map_name = os.getenv("PLATOON_MAP", self._stack_config.get("map", ""))
        self.world = self._connect_world(map_name)
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(
            os.getenv("PLATOON_DT", str(self._stack_config.get("fixed_delta_seconds", 0.05))),
        )
        settings.substepping = True
        settings.max_substep_delta_time = min(float(os.getenv("PLATOON_SUBSTEP_DT", "0.01")), settings.fixed_delta_seconds)
        settings.max_substeps = int(os.getenv("PLATOON_MAX_SUBSTEPS", "10"))
        self.world.apply_settings(settings)
        self.dt = float(settings.fixed_delta_seconds)

        self.h = float(os.getenv("PLATOON_HEADWAY", str(self._stack_config.get("headway", 1.0))))
        self.d_0 = float(os.getenv("PLATOON_STANDSTILL_DIST", str(self._stack_config.get("standstill_distance", 7.0))))
        self.leader_throttle = float(os.getenv("PLATOON_LEADER_THROTTLE", str(self._stack_config.get("leader_throttle", 0.35))))
        self.vehicle_gap = float(os.getenv("PLATOON_INITIAL_GAP", str(self._stack_config.get("initial_gap", 15.0))))
        self.vehicle_specs = list(self._stack_config.get("vehicles", []))
        self._leader_speed_candidates = [15.0, 17.5, 20.0]
        self._rng = np.random.default_rng(int(os.getenv("PLATOON_PROFILE_SEED", "0")) or None)
        self._leader_profile = self._sample_leader_profile()
        self._leader_profile_step = 0
        self._reset_requested = False
        self._last_reset_time = 0.0
        self._reset_cooldown_sec = float(os.getenv("PLATOON_RESET_COOLDOWN_SEC", "2.0"))
        self._lateral_lookahead = float(os.getenv("PLATOON_LAT_LOOKAHEAD", "6.0"))
        self._lat_pid_args = {
            "K_P": float(os.getenv("PLATOON_LAT_KP", "1.95")),
            "K_I": float(os.getenv("PLATOON_LAT_KI", "0.05")),
            "K_D": float(os.getenv("PLATOON_LAT_KD", "0.2")),
            "dt": self.dt,
        }
        self._control_timeout_sec = float(os.getenv("PLATOON_CONTROL_TIMEOUT_SEC", "0.2"))
        self._fallback_kp = float(os.getenv("PLATOON_FALLBACK_KP", "0.12"))
        self._fallback_kd = float(os.getenv("PLATOON_FALLBACK_KD", "0.35"))
        self._fallback_max_throttle = float(os.getenv("PLATOON_FALLBACK_MAX_THROTTLE", "0.55"))
        self._fallback_max_brake = float(os.getenv("PLATOON_FALLBACK_MAX_BRAKE", "0.65"))
        self._rl_blend_alpha = float(os.getenv("PLATOON_RL_BLEND_ALPHA", "1.0"))
        self._spectator_follow = os.getenv("PLATOON_SPECTATOR_FOLLOW", "1") == "1"
        self._spectator_distance = float(os.getenv("PLATOON_SPECTATOR_DISTANCE", "12.0"))
        self._spectator_height = float(os.getenv("PLATOON_SPECTATOR_HEIGHT", "5.0"))
        self._spectator_pitch = float(os.getenv("PLATOON_SPECTATOR_PITCH", "-15.0"))

        self.vehicles: list[carla.Actor] = []
        self._lateral_controllers: list[PIDLateralController] = []
        self._last_follower_controls: list[dict[str, float]] = [
            {"throttle": 0.0, "brake": 0.0, "timestamp": 0.0} for _ in range(3)
        ]
        self._spawn_platoon()

        self.control_subs = []
        self.state_pubs = []
        self.reset_sub = self.create_subscription(
            Empty,
            "/carla/platoon/reset",
            self._handle_reset,
            10,
        )
        for i in range(3):
            vehicle_id = self._follower_vehicle_id(i)
            sub = self.create_subscription(
                Float32MultiArray,
                f"/carla/{vehicle_id}/vehicle_control",
                lambda msg, idx=i: self.apply_control(idx, msg),
                10,
            )
            self.control_subs.append(sub)

            pub = self.create_publisher(
                Float32MultiArray,
                f"/carla/{vehicle_id}/vehicle_state",
                10,
            )
            self.state_pubs.append(pub)

        self.get_logger().info(f"Connected to CARLA map: {self.world.get_map().name}")

    def _load_stack_config(self) -> dict:
        config_path = os.getenv(
            "PLATOON_STACK_CONFIG",
            os.path.join(os.path.dirname(__file__), "config", "platoon_stack.json"),
        )
        if not os.path.exists(config_path):
            return {}
        with open(config_path, encoding="utf-8") as file:
            return json.load(file)

    def _connect_world(self, map_name: str) -> carla.World:
        retries = int(os.getenv("CARLA_STARTUP_RETRIES", "20"))
        retry_delay = float(os.getenv("CARLA_STARTUP_RETRY_SEC", "2.0"))
        last_error: Exception | None = None
        for attempt in range(retries):
            try:
                if map_name:
                    return self.client.load_world(map_name)
                return self.client.get_world()
            except RuntimeError as error:
                last_error = error
                if attempt == retries - 1:
                    break
                time.sleep(retry_delay)
        raise RuntimeError(f"Failed to connect to CARLA world after {retries} attempts: {last_error}")

    def _sample_leader_profile(self) -> dict[str, float]:
        target_speed = float(self._rng.choice(self._leader_speed_candidates))
        return {
            "target_speed": target_speed,
            "cruise_steps": int(os.getenv("PLATOON_CRUISE_STEPS", "120")),
            "brake_steps": int(os.getenv("PLATOON_BRAKE_STEPS", "40")),
        }

    def _handle_reset(self, _: Empty) -> None:
        now = time.monotonic()
        if (now - self._last_reset_time) < self._reset_cooldown_sec:
            self.get_logger().info("Ignoring duplicate reset request inside cooldown window")
            return
        self._last_reset_time = now
        self._reset_requested = True

    def _follower_vehicle_id(self, follower_idx: int) -> str:
        if self.vehicle_specs and (follower_idx + 1) < len(self.vehicle_specs):
            return str(self.vehicle_specs[follower_idx + 1].get("id", f"follower_{follower_idx + 1}"))
        return f"follower_{follower_idx + 1}"

    def _vehicle_blueprint(self, vehicle_cfg: dict) -> carla.ActorBlueprint:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_type = vehicle_cfg.get("type", "vehicle.tesla.model3")
        matches = blueprint_library.filter(vehicle_type)
        if not matches:
            matches = blueprint_library.filter("vehicle.*model3*")
        if not matches:
            matches = blueprint_library.filter("vehicle.*")
        blueprint = matches[0]
        if vehicle_cfg.get("role_name"):
            blueprint.set_attribute("role_name", str(vehicle_cfg["role_name"]))
        if vehicle_cfg.get("ros_name"):
            blueprint.set_attribute("ros_name", str(vehicle_cfg["ros_name"]))
        return blueprint

    @staticmethod
    def _transform_from_json(spawn_point: dict) -> carla.Transform:
        return carla.Transform(
            location=carla.Location(
                x=float(spawn_point["x"]),
                y=-float(spawn_point["y"]),
                z=float(spawn_point["z"]),
            ),
            rotation=carla.Rotation(
                roll=float(spawn_point.get("roll", 0.0)),
                pitch=-float(spawn_point.get("pitch", 0.0)),
                yaw=-float(spawn_point.get("yaw", 0.0)),
            ),
        )

    @staticmethod
    def _yaw_diff_deg(lhs: float, rhs: float) -> float:
        return abs((lhs - rhs + 180.0) % 360.0 - 180.0)

    def _candidate_platoon_transforms(self, base_transform: carla.Transform) -> list[carla.Transform] | None:
        road_map = self.world.get_map()
        waypoint = road_map.get_waypoint(
            base_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None or waypoint.is_junction:
            return None

        base_road_id = waypoint.road_id
        base_lane_id = waypoint.lane_id
        base_yaw = waypoint.transform.rotation.yaw

        transforms = []
        current_waypoint = waypoint
        for index in range(4):
            if current_waypoint.is_junction:
                return None
            if current_waypoint.road_id != base_road_id or current_waypoint.lane_id != base_lane_id:
                return None
            if self._yaw_diff_deg(current_waypoint.transform.rotation.yaw, base_yaw) > 8.0:
                return None

            transform = carla.Transform(current_waypoint.transform.location, current_waypoint.transform.rotation)
            transform.location.z += 0.2
            transforms.append(transform)
            if index == 3:
                break

            previous_waypoints = [
                wp for wp in current_waypoint.previous(self.vehicle_gap)
                if not wp.is_junction
                and wp.road_id == base_road_id
                and wp.lane_id == base_lane_id
                and self._yaw_diff_deg(wp.transform.rotation.yaw, base_yaw) <= 8.0
            ]
            if not previous_waypoints:
                return None
            current_waypoint = min(
                previous_waypoints,
                key=lambda wp: self._yaw_diff_deg(wp.transform.rotation.yaw, base_yaw),
            )

        return transforms

    def _destroy_actors(self, actors: Iterable[carla.Actor]) -> None:
        batch = [carla.command.DestroyActor(actor) for actor in actors if actor is not None and actor.is_alive]
        if batch:
            self.get_logger().info(f"Destroying {len(batch)} actors")
            self.client.apply_batch_sync(batch, True)

    def _expected_role_names(self) -> set[str]:
        expected_roles = {cfg.get("role_name") for cfg in self.vehicle_specs if cfg.get("role_name")}
        return expected_roles or {"leader", "follower_1", "follower_2", "follower_3"}

    def _cleanup_existing_platoon_actors(self) -> None:
        expected_roles = self._expected_role_names()
        for _ in range(3):
            self.world.tick()
            existing = []
            for actor in self.world.get_actors().filter("vehicle.*"):
                if actor.attributes.get("role_name") in expected_roles and actor.is_alive:
                    existing.append(actor)
            if not existing:
                return
            self.get_logger().info(f"Cleaning up {len(existing)} existing platoon actors before spawn")
            self._destroy_actors(existing)
        self.world.tick()

    def _validate_unique_platoon_actors(self) -> None:
        expected_roles = self._expected_role_names()
        self.world.tick()
        role_counts: dict[str, int] = {}
        for actor in self.world.get_actors().filter("vehicle.*"):
            role = actor.attributes.get("role_name")
            if role in expected_roles and actor.is_alive:
                role_counts[role] = role_counts.get(role, 0) + 1
        duplicates = {role: count for role, count in role_counts.items() if count > 1}
        if duplicates:
            raise RuntimeError(f"Duplicate platoon actors detected after spawn: {duplicates}")

    def _build_lateral_controllers(self) -> None:
        self._lateral_controllers = [
            PIDLateralController(vehicle, offset=0, **self._lat_pid_args)
            for vehicle in self.vehicles
        ]

    def _steer_for_vehicle(self, vehicle: carla.Vehicle, controller_idx: int) -> float:
        if controller_idx >= len(self._lateral_controllers):
            return 0.0
        target_wp = self._target_waypoint(vehicle)
        if target_wp is None:
            return 0.0
        steer = self._lateral_controllers[controller_idx].run_step(target_wp)
        return float(np.clip(steer, -0.8, 0.8))

    def _target_waypoint(self, vehicle: carla.Vehicle) -> carla.Waypoint | None:
        waypoint = self.world.get_map().get_waypoint(
            vehicle.get_location(),
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None
        forward_candidates = waypoint.next(self._lateral_lookahead)
        return forward_candidates[0] if forward_candidates else waypoint

    def _update_spectator(self, vehicle: carla.Vehicle) -> None:
        if not self._spectator_follow or vehicle is None or not vehicle.is_alive:
            return
        transform = vehicle.get_transform()
        yaw_rad = math.radians(transform.rotation.yaw)
        offset = carla.Location(
            x=-math.cos(yaw_rad) * self._spectator_distance,
            y=-math.sin(yaw_rad) * self._spectator_distance,
            z=self._spectator_height,
        )
        spectator_transform = carla.Transform(
            transform.location + offset,
            carla.Rotation(
                pitch=self._spectator_pitch,
                yaw=transform.rotation.yaw,
                roll=0.0,
            ),
        )
        self.world.get_spectator().set_transform(spectator_transform)

    def _spawn_platoon(self) -> None:
        self.get_logger().info("Spawning platoon vehicles")
        self._cleanup_existing_platoon_actors()
        if self.vehicle_specs and all("spawn_point" in vehicle_cfg for vehicle_cfg in self.vehicle_specs):
            self._spawn_platoon_from_config()
            return

        spawn_points = list(self.world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        candidates: list[tuple[float, list[carla.Transform]]] = []
        for base_transform in spawn_points:
            transforms = self._candidate_platoon_transforms(base_transform)
            if transforms is None:
                continue
            yaws = [transform.rotation.yaw for transform in transforms]
            score = sum(self._yaw_diff_deg(yaw, yaws[0]) for yaw in yaws)
            candidates.append((score, transforms))

        for _, transforms in sorted(candidates, key=lambda item: item[0]):
            commands = []
            for index, transform in enumerate(transforms):
                vehicle_cfg = self.vehicle_specs[index] if index < len(self.vehicle_specs) else {}
                commands.append(carla.command.SpawnActor(self._vehicle_blueprint(vehicle_cfg), transform))
            responses = self.client.apply_batch_sync(commands, True)
            actor_ids: list[int] = []
            spawn_failed = False
            for response in responses:
                if response.error:
                    self.get_logger().warning(f"Spawn response error: {response.error}")
                    spawn_failed = True
                    break
                actor_ids.append(response.actor_id)

            if spawn_failed:
                if actor_ids:
                    self.client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actor_ids], True)
                continue

            actors = [self.world.get_actor(actor_id) for actor_id in actor_ids]
            if all(actor is not None for actor in actors):
                self.vehicles = [actor for actor in actors if actor is not None]
                for vehicle in self.vehicles:
                    vehicle.set_autopilot(False)
                self._build_lateral_controllers()
                self.get_logger().info(f"Spawned platoon with {len(self.vehicles)} vehicles on inferred lane")
                return

        raise RuntimeError("Failed to spawn full platoon on a single straight driving lane")

    def _spawn_platoon_from_config(self) -> None:
        commands = []
        for vehicle_cfg in self.vehicle_specs:
            transform = self._transform_from_json(vehicle_cfg["spawn_point"])
            self.get_logger().info(
                f"Spawn request role={vehicle_cfg.get('role_name')} x={transform.location.x:.2f} y={transform.location.y:.2f} z={transform.location.z:.2f} yaw={transform.rotation.yaw:.2f}",
            )
            commands.append(carla.command.SpawnActor(self._vehicle_blueprint(vehicle_cfg), transform))

        responses = self.client.apply_batch_sync(commands, True)
        actor_ids: list[int] = []
        for response in responses:
            if response.error:
                self.get_logger().warning(f"Spawn response error: {response.error}")
                if actor_ids:
                    self.client.apply_batch_sync([carla.command.DestroyActor(actor_id) for actor_id in actor_ids], True)
                raise RuntimeError(f"Failed to spawn platoon from config: {response.error}")
            actor_ids.append(response.actor_id)

        actors = [self.world.get_actor(actor_id) for actor_id in actor_ids]
        self.vehicles = [actor for actor in actors if actor is not None]
        for vehicle in self.vehicles:
            vehicle.set_autopilot(False)
            transform = vehicle.get_transform()
            self.get_logger().info(
                f"Spawn result role={vehicle.attributes.get('role_name')} x={transform.location.x:.2f} y={transform.location.y:.2f} z={transform.location.z:.2f} yaw={transform.rotation.yaw:.2f}",
            )
        self._build_lateral_controllers()
        self._validate_unique_platoon_actors()
        self.get_logger().info(f"Spawned platoon from config with {len(self.vehicles)} vehicles")

    def apply_control(self, follower_idx: int, msg: Float32MultiArray) -> None:
        throttle, brake, _ = list(msg.data)[:3]
        if follower_idx < len(self._last_follower_controls):
            self._last_follower_controls[follower_idx] = {
                "throttle": float(np.clip(throttle, 0.0, 1.0)),
                "brake": float(np.clip(brake, 0.0, 1.0)),
                "timestamp": time.monotonic(),
            }

    def _fallback_longitudinal_control(self, preceding_vehicle: carla.Vehicle, ego_vehicle: carla.Vehicle) -> tuple[float, float]:
        preceding_velocity = _velocity_norm(preceding_vehicle.get_velocity())
        ego_velocity = _velocity_norm(ego_vehicle.get_velocity())
        actual_distance = self._pathwise_distance(preceding_vehicle.get_location(), ego_vehicle.get_location())
        desired_distance = ego_velocity * self.h + self.d_0
        spacing_error = actual_distance - desired_distance
        rel_vel = preceding_velocity - ego_velocity
        accel_cmd = self._fallback_kp * spacing_error + self._fallback_kd * rel_vel
        if ego_velocity < 0.5 and preceding_velocity > 1.0 and spacing_error > 0.5:
            accel_cmd = max(accel_cmd, 1.2)
        throttle = float(np.clip(max(accel_cmd, 0.0) / 3.0, 0.0, self._fallback_max_throttle))
        brake = float(np.clip(max(-accel_cmd, 0.0) / 4.0, 0.0, self._fallback_max_brake))
        return throttle, brake

    def tick_simulation(self) -> None:
        if self._shutdown_requested:
            return
        if self._reset_requested:
            self.get_logger().info("Reset requested, respawning platoon")
            self._destroy_actors(self.vehicles)
            self.vehicles.clear()
            self._spawn_platoon()
            self._leader_profile = self._sample_leader_profile()
            self._leader_profile_step = 0
            self._last_follower_controls = [
                {"throttle": 0.0, "brake": 0.0, "timestamp": 0.0} for _ in range(3)
            ]
            self._reset_requested = False

        if not self.vehicles:
            self.get_logger().warning("No vehicles in bridge state, attempting respawn")
            self._spawn_platoon()
            if not self.vehicles:
                return
        elif not all(vehicle is not None and vehicle.is_alive for vehicle in self.vehicles):
            self.get_logger().warning("Detected missing/dead vehicle, respawning platoon")
            self._destroy_actors(self.vehicles)
            self.vehicles.clear()
            self._spawn_platoon()
            if not self.vehicles:
                return

        if len(self.vehicles) < 4:
            self.get_logger().warning(f"Incomplete platoon ({len(self.vehicles)} vehicles), respawning")
            self._destroy_actors(self.vehicles)
            self.vehicles.clear()
            self._spawn_platoon()
            if len(self.vehicles) < 4:
                return

        if self._leader_profile_step == 0:
            self.get_logger().info(
                f"Leader profile target={self._leader_profile['target_speed']} cruise_steps={self._leader_profile['cruise_steps']} brake_steps={self._leader_profile['brake_steps']}",
            )

        if not self.vehicles:
            return
        leader_vehicle = self.vehicles[0]
        leader_speed = _velocity_norm(leader_vehicle.get_velocity())
        target_speed = float(self._leader_profile["target_speed"])
        cruise_steps = int(self._leader_profile["cruise_steps"])
        brake_steps = int(self._leader_profile["brake_steps"])
        profile_total_steps = 200 + cruise_steps + brake_steps
        if self._leader_profile_step >= profile_total_steps and leader_speed < 0.5:
            self._leader_profile = self._sample_leader_profile()
            self._leader_profile_step = 0
            target_speed = float(self._leader_profile["target_speed"])
            cruise_steps = int(self._leader_profile["cruise_steps"])
            brake_steps = int(self._leader_profile["brake_steps"])
            profile_total_steps = 200 + cruise_steps + brake_steps
            self.get_logger().info(
                f"Leader profile resampled target={self._leader_profile['target_speed']} cruise_steps={self._leader_profile['cruise_steps']} brake_steps={self._leader_profile['brake_steps']}",
            )
        if leader_speed < target_speed and self._leader_profile_step < 200:
            throttle = self.leader_throttle
            brake = 0.0
        elif self._leader_profile_step < 200 + cruise_steps:
            throttle = 0.15
            brake = 0.0
        elif self._leader_profile_step < 200 + cruise_steps + brake_steps:
            throttle = 0.0
            brake = 0.6
        else:
            throttle = 0.0
            brake = 0.0
        leader_control = carla.VehicleControl()
        leader_control.steer = self._steer_for_vehicle(leader_vehicle, 0)
        leader_control.throttle = float(np.clip(throttle, 0.0, 0.75))
        leader_control.brake = float(np.clip(brake, 0.0, 0.8))
        leader_control.hand_brake = False
        leader_control.manual_gear_shift = False
        leader_vehicle.apply_control(leader_control)
        for i in range(3):
            vehicle = self.vehicles[i + 1]
            follower_control = carla.VehicleControl()
            follower_control.steer = self._steer_for_vehicle(vehicle, i + 1)
            control_state = self._last_follower_controls[i]
            fallback_throttle, fallback_brake = self._fallback_longitudinal_control(self.vehicles[i], vehicle)
            if (time.monotonic() - control_state["timestamp"]) <= self._control_timeout_sec:
                throttle_cmd = ((1.0 - self._rl_blend_alpha) * fallback_throttle) + (
                    self._rl_blend_alpha * control_state["throttle"]
                )
                brake_cmd = ((1.0 - self._rl_blend_alpha) * fallback_brake) + (
                    self._rl_blend_alpha * control_state["brake"]
                )
            else:
                throttle_cmd, brake_cmd = fallback_throttle, fallback_brake
            follower_control.throttle = float(np.clip(throttle_cmd, 0.0, 1.0))
            follower_control.brake = float(np.clip(brake_cmd, 0.0, 1.0))
            follower_control.hand_brake = False
            follower_control.manual_gear_shift = False
            vehicle.apply_control(follower_control)
        self.world.tick()
        self._update_spectator(leader_vehicle)
        self._leader_profile_step += 1
        for i in range(3):
            self.publish_state(i)

    def _pathwise_distance(self, preceding_location: carla.Location, ego_location: carla.Location) -> float:
        preceding_wp = self.world.get_map().get_waypoint(preceding_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        ego_wp = self.world.get_map().get_waypoint(ego_location, project_to_road=True, lane_type=carla.LaneType.Driving)
        if preceding_wp and ego_wp:
            if preceding_wp.road_id == ego_wp.road_id and preceding_wp.lane_id == ego_wp.lane_id:
                return float(abs(preceding_wp.s - ego_wp.s))
        return float(preceding_location.distance(ego_location))

    def publish_state(self, follower_idx: int) -> None:
        preceding_vehicle = self.vehicles[follower_idx]
        ego_vehicle = self.vehicles[follower_idx + 1]

        preceding_velocity = _velocity_norm(preceding_vehicle.get_velocity())
        ego_velocity = _velocity_norm(ego_vehicle.get_velocity())
        preceding_accel = _velocity_norm(preceding_vehicle.get_acceleration())

        preceding_location = preceding_vehicle.get_location()
        ego_location = ego_vehicle.get_location()
        actual_distance = self._pathwise_distance(preceding_location, ego_location)
        desired_distance = ego_velocity * self.h + self.d_0
        spacing_error = actual_distance - desired_distance
        rel_vel = preceding_velocity - ego_velocity

        waypoint = self.world.get_map().get_waypoint(
            ego_location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        lat_offset = ego_location.distance(waypoint.transform.location) if waypoint else 0.0

        msg = Float32MultiArray()
        msg.data = [
            float(spacing_error),
            float(rel_vel),
            float(preceding_accel),
            float(ego_velocity),
            float(lat_offset),
        ]
        try:
            self.state_pubs[follower_idx].publish(msg)
        except RCLError:
            self._shutdown_requested = True

    def close(self) -> None:
        self._destroy_actors(self.vehicles)
        self.vehicles.clear()
        self.world.apply_settings(self.original_settings)


def main() -> None:
    rclpy.init()
    bridge = CarlaPlatoonBridge()
    try:
        while rclpy.ok() and not bridge._shutdown_requested:
            try:
                rclpy.spin_once(bridge, timeout_sec=0.01)
                bridge.tick_simulation()
            except ExternalShutdownException:
                break
            except RCLError:
                break
            except Exception as exc:
                message = str(exc).lower()
                if "context is invalid" in message or "rcl_shutdown" in message or "wait set" in message:
                    break
                raise
    except KeyboardInterrupt:
        pass
    finally:
        try:
            bridge.close()
        except Exception:
            pass
        try:
            bridge.destroy_node()
        except Exception:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
