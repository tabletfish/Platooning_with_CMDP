import carla
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
import numpy as np

class CarlaPlatoonBridge(Node):
    def __init__(self):
        super().__init__('carla_platoon_bridge')
        
        # 1. CARLA 클라이언트 접속 및 동기화 모드 설정
        self.client = carla.Client('localhost', 2000)
        self.client.set_timeout(10.0)
        self.world = self.client.load_world('Town04') # 논문 평가 환경 
        
        settings = self.world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05 # 0.05초 동기화 틱 
        self.world.apply_settings(settings)
        
        # 2. 차량 소환 (Leader 1대 + Follower 3대)
        self.vehicles = []
        self._spawn_platoon()
        
        # 3. ROS2 구독(Subscriber) 및 발행(Publisher) 설정
        self.control_subs = []
        self.state_pubs = []
        
        for i in range(3): # Follower 1, 2, 3
            # RL 환경에서 오는 제어 명령 수신
            sub = self.create_subscription(
                Float32MultiArray,
                f'/carla/follower_{i+1}/vehicle_control',
                lambda msg, idx=i: self.apply_control(idx, msg),
                10
            )
            self.control_subs.append(sub)
            
            # RL 환경으로 차량 상태 전송
            pub = self.create_publisher(
                Float32MultiArray,
                f'/carla/follower_{i+1}/vehicle_state',
                10
            )
            self.state_pubs.append(pub)

    def _spawn_platoon(self):
        # CARLA 블루프린트 라이브러리에서 트럭 모델 로드 및 일렬로 소환하는 로직
        blueprint_library = self.world.get_blueprint_library()
        truck_bp = blueprint_library.filter('vehicle.carlamotors.carlacola')[0]
        spawn_points = self.world.get_map().get_spawn_points()
        
        # 앞차부터 순서대로 간격을 두고 소환 (Leader -> Follower 1 -> 2 -> 3)
        for i in range(4):
            transform = spawn_points[0] # 임의의 시작점 (실제 구현시 좌표 조정 필요)
            transform.location.x -= i * 15.0 # 15m 간격
            vehicle = self.world.spawn_actor(truck_bp, transform)
            self.vehicles.append(vehicle)
            
    def apply_control(self, follower_idx, msg):
        # 수신된 제어 명령(Throttle, Brake)을 CARLA 차량에 적용
        throttle, brake, _ = msg.data
        vehicle = self.vehicles[follower_idx + 1] # 0번은 Leader
        
        control = carla.VehicleControl()
        control.throttle = float(throttle)
        control.brake = float(brake)
        vehicle.apply_control(control)

    def tick_simulation(self):
        # CARLA 월드 1스텝(0.05초) 진행
        self.world.tick()
        
        # 각 Follower의 상태(간격 오차, 상대 속도 등)를 계산하여 ROS2로 발행
        for i in range(3):
            self.publish_state(i)

    def publish_state(self, follower_idx):
        leader_vehicle = self.vehicles[follower_idx]      # i-1 번째 차량
        ego_vehicle = self.vehicles[follower_idx + 1]     # i 번째 차량
        
        # 물리 엔진에서 속도 및 위치 정보 추출하여 Delta d, Delta v 등 계산
        # (상세 수학적 거리 계산식은 생략)
        spacing_error = 0.0 # d_i(t) - d_i^{CTG}(t) 
        rel_vel = 0.0       # v_{i-1}(t) - v_i(t) 
        prec_accel = 0.0
        ego_vel = ego_vehicle.get_velocity().x
        lat_offset = 0.0
        
        msg = Float32MultiArray()
        msg.data = [spacing_error, rel_vel, prec_accel, ego_vel, lat_offset]
        self.state_pubs[follower_idx].publish(msg)

def main():
    rclpy.init()
    bridge = CarlaPlatoonBridge()
    
    try:
        while rclpy.ok():
            rclpy.spin_once(bridge, timeout_sec=0.01)
            bridge.tick_simulation()
    except KeyboardInterrupt:
        pass
    finally:
        bridge.client.get_world().apply_settings(carla.WorldSettings(synchronous_mode=False))
        rclpy.shutdown()

if __name__ == '__main__':
    main()
