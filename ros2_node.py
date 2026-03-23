import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
# 실제 CARLA ROS2 Bridge를 사용할 경우 carla_msgs 등을 사용합니다.
# 여기서는 데이터 구조를 직관적으로 보여주기 위해 임의의 메시지 타입을 가정합니다.

import time

class PlatoonROS2Node(Node):
    def __init__(self):
        super().__init__('platoon_rl_controller_node')
        
        # 1. Publisher: CARLA로 제어 명령 (Throttle, Brake) 전송
        self.control_pub = self.create_publisher(
            Float32MultiArray, 
            '/carla/follower_1/vehicle_control', 
            10
        )
        
        # 2. Subscriber: CARLA로부터 센서 및 차량 상태 데이터 수신
        self.state_sub = self.create_subscription(
            Float32MultiArray,
            '/carla/follower_1/vehicle_state',
            self.state_callback,
            10
        )
        
        # 동기화 제어를 위한 상태 저장 버퍼
        self.latest_data = {
            'spacing_error': 0.0,
            'rel_vel': 0.0,
            'prec_accel': 0.0,
            'ego_vel': 0.0,
            'lat_offset': 0.0
        }
        self.data_received = False

    def publish_control(self, throttle, brake):
        """환경 클래스에서 계산된 Throttle과 Brake를 ROS2 토픽으로 발행"""
        msg = Float32MultiArray()
        # 제어 명령 배열: [throttle, brake, steer(생략가능)]
        msg.data = [float(throttle), float(brake), 0.0] 
        self.control_pub.publish(msg)

    def state_callback(self, msg):
        """CARLA에서 차량 상태 토픽이 들어올 때마다 버퍼 업데이트"""
        # msg.data 구성 가정: [spacing_error, rel_vel, prec_accel, ego_vel, lat_offset]
        data = msg.data
        if len(data) >= 5:
            self.latest_data['spacing_error'] = data[0]
            self.latest_data['rel_vel'] = data[1]
            self.latest_data['prec_accel'] = data[2]
            self.latest_data['ego_vel'] = data[3]
            self.latest_data['lat_offset'] = data[4]
            
        self.data_received = True

    def get_latest_data(self):
        """환경 클래스가 최신 상태값을 요청할 때 반환"""
        return self.latest_data

    def tick_and_wait(self, dt=0.05):
        """
        논문에 명시된 0.05초(Synchronous Tick) 주기를 맞추기 위한 함수.
        새로운 상태 데이터가 들어올 때까지 ROS2 이벤트를 처리하며 대기합니다.
        """
        self.data_received = False
        start_time = time.time()
        
        # 새로운 콜백(데이터 수신)이 발생하거나, timeout(dt)이 지날 때까지 대기
        while not self.data_received and (time.time() - start_time) < dt:
            rclpy.spin_once(self, timeout_sec=0.01)
            
        # 동기화 시뮬레이션 환경이므로 필요에 따라 CARLA 서버에 tick 명령을 여기서 발행할 수도 있습니다.