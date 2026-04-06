from __future__ import annotations

import math
import os

import carla
import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


def _velocity_norm(vector: carla.Vector3D) -> float:
    return float(math.sqrt(vector.x**2 + vector.y**2 + vector.z**2))


class CarlaPlatoonBridge(Node):
    def __init__(self) -> None:
        super().__init__("carla_platoon_bridge")

        self.client = carla.Client(os.getenv("CARLA_HOST", "localhost"), int(os.getenv("CARLA_PORT", "2000")))
        self.client.set_timeout(float(os.getenv("CARLA_TIMEOUT", "10.0")))

        map_name = os.getenv("PLATOON_MAP", "")
        self.world = self.client.load_world(map_name) if map_name else self.client.get_world()
        self.original_settings = self.world.get_settings()

        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = float(os.getenv("PLATOON_DT", "0.05"))
        self.world.apply_settings(settings)
        self.dt = float(settings.fixed_delta_seconds)

        self.h = float(os.getenv("PLATOON_HEADWAY", "1.0"))
        self.d_0 = float(os.getenv("PLATOON_STANDSTILL_DIST", "7.0"))
        self.leader_throttle = float(os.getenv("PLATOON_LEADER_THROTTLE", "0.35"))
        self.vehicle_gap = float(os.getenv("PLATOON_INITIAL_GAP", "15.0"))

        self.vehicles: list[carla.Actor] = []
        self._spawn_platoon()

        self.control_subs = []
        self.state_pubs = []
        for i in range(3):
            sub = self.create_subscription(
                Float32MultiArray,
                f"/carla/follower_{i + 1}/vehicle_control",
                lambda msg, idx=i: self.apply_control(idx, msg),
                10,
            )
            self.control_subs.append(sub)

            pub = self.create_publisher(
                Float32MultiArray,
                f"/carla/follower_{i + 1}/vehicle_state",
                10,
            )
            self.state_pubs.append(pub)

        self.get_logger().info(f"Connected to CARLA map: {self.world.get_map().name}")

    def _candidate_platoon_transforms(self, base_transform: carla.Transform) -> list[carla.Transform] | None:
        waypoint = self.world.get_map().get_waypoint(
            base_transform.location,
            project_to_road=True,
            lane_type=carla.LaneType.Driving,
        )
        if waypoint is None:
            return None

        transforms = []
        current_waypoint = waypoint
        for index in range(4):
            transform = carla.Transform(current_waypoint.transform.location, current_waypoint.transform.rotation)
            transform.location.z += 0.2
            transforms.append(transform)
            if index == 3:
                break
            previous_waypoints = current_waypoint.previous(self.vehicle_gap)
            if not previous_waypoints:
                return None
            current_waypoint = previous_waypoints[0]
        return transforms

    def _spawn_platoon(self) -> None:
        blueprint_library = self.world.get_blueprint_library()
        vehicle_bp = blueprint_library.filter("vehicle.*model3*")
        if not vehicle_bp:
            vehicle_bp = blueprint_library.filter("vehicle.*")
        ego_bp = vehicle_bp[0]
        spawn_points = list(self.world.get_map().get_spawn_points())
        if not spawn_points:
            raise RuntimeError("No spawn points available in CARLA map")

        for base_transform in spawn_points:
            transforms = self._candidate_platoon_transforms(base_transform)
            if transforms is None:
                continue

            spawned: list[carla.Actor] = []
            for transform in transforms:
                vehicle = self.world.try_spawn_actor(ego_bp, transform)
                if vehicle is None:
                    for actor in spawned:
                        if actor.is_alive:
                            actor.destroy()
                    spawned = []
                    break
                vehicle.set_autopilot(False)
                spawned.append(vehicle)

            if len(spawned) == 4:
                self.vehicles = spawned
                return

        raise RuntimeError("Failed to spawn full platoon on a single driving lane")

    def apply_control(self, follower_idx: int, msg: Float32MultiArray) -> None:
        throttle, brake, _ = list(msg.data)[:3]
        vehicle = self.vehicles[follower_idx + 1]

        control = carla.VehicleControl()
        control.throttle = float(np.clip(throttle, 0.0, 1.0))
        control.brake = float(np.clip(brake, 0.0, 1.0))
        vehicle.apply_control(control)

    def tick_simulation(self) -> None:
        leader_control = carla.VehicleControl(throttle=self.leader_throttle, brake=0.0)
        self.vehicles[0].apply_control(leader_control)
        self.world.tick()
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
        self.state_pubs[follower_idx].publish(msg)

    def close(self) -> None:
        for vehicle in self.vehicles:
            if vehicle.is_alive:
                vehicle.destroy()
        self.vehicles.clear()
        self.world.apply_settings(self.original_settings)


def main() -> None:
    rclpy.init()
    bridge = CarlaPlatoonBridge()
    try:
        while rclpy.ok():
            rclpy.spin_once(bridge, timeout_sec=0.01)
            bridge.tick_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.close()
        bridge.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
