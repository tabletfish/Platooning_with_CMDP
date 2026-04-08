from __future__ import annotations

import argparse
import math
import time

import carla


def _speed(actor: carla.Actor) -> float:
    velocity = actor.get_velocity()
    return math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2)


def _distance(a: carla.Actor, b: carla.Actor) -> float:
    return a.get_location().distance(b.get_location())


def _vehicle_by_role(world: carla.World, role_name: str) -> carla.Actor | None:
    for actor in world.get_actors().filter("vehicle.*"):
        if actor.attributes.get("role_name") == role_name:
            return actor
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=2000)
    parser.add_argument("--samples", type=int, default=60)
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--vehicle-length", type=float, default=4.8)
    args = parser.parse_args()

    client = carla.Client(args.host, args.port)
    client.set_timeout(5.0)
    world = client.get_world()
    print(f"map={world.get_map().name}")

    for sample in range(args.samples):
        leader = _vehicle_by_role(world, "leader")
        follower = _vehicle_by_role(world, "follower_1")
        if leader is None or follower is None:
            print(f"{sample:04d} missing leader/follower actors")
            time.sleep(args.interval)
            continue

        leader_speed = _speed(leader)
        follower_speed = _speed(follower)
        center_distance = _distance(leader, follower)
        clearance = max(center_distance - args.vehicle_length, 0.0)
        thw = clearance / follower_speed if follower_speed > 0.1 else float("inf")
        thw_text = f"{thw:6.2f}" if math.isfinite(thw) else "   inf"
        print(
            f"{sample:04d} leader={leader_speed:5.2f}m/s "
            f"f1={follower_speed:5.2f}m/s "
            f"clearance={clearance:6.2f}m thw={thw_text}s",
        )
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
