from __future__ import annotations

import os
import time

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


class PlatoonROS2Node(Node):
    def __init__(self, follower_id: int = 1):
        node_name = f"platoon_rl_controller_node_{os.getpid()}_{follower_id}"
        super().__init__(node_name)

        topic_prefix = f"/carla/follower_{follower_id}"
        self.control_pub = self.create_publisher(
            Float32MultiArray,
            f"{topic_prefix}/vehicle_control",
            10,
        )
        self.state_sub = self.create_subscription(
            Float32MultiArray,
            f"{topic_prefix}/vehicle_state",
            self.state_callback,
            10,
        )

        self.latest_data = {
            "spacing_error": 0.0,
            "rel_vel": 0.0,
            "prec_accel": 0.0,
            "ego_vel": 0.0,
            "lat_offset": 0.0,
        }
        self.data_received = False

    def publish_control(self, throttle: float, brake: float) -> None:
        msg = Float32MultiArray()
        msg.data = [float(throttle), float(brake), 0.0]
        self.control_pub.publish(msg)

    def state_callback(self, msg: Float32MultiArray) -> None:
        data = list(msg.data)
        if len(data) >= 5:
            self.latest_data["spacing_error"] = data[0]
            self.latest_data["rel_vel"] = data[1]
            self.latest_data["prec_accel"] = data[2]
            self.latest_data["ego_vel"] = data[3]
            self.latest_data["lat_offset"] = data[4]
        self.data_received = True

    def get_latest_data(self) -> dict[str, float]:
        return dict(self.latest_data)

    def tick_and_wait(self, dt: float = 0.05) -> bool:
        self.data_received = False
        start_time = time.time()
        while not self.data_received and (time.time() - start_time) < dt:
            rclpy.spin_once(self, timeout_sec=0.01)
        return self.data_received
