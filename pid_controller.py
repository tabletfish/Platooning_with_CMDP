import numpy as np

class LongitudinalPIDController:
    def __init__(self, dt=0.05):
        # 논문의 플래툰 PID 제어에 일반적으로 사용되는 게인 값 (임의 튜닝값)
        self.kp_d = 0.5   # 거리 오차 비례 게인
        self.kd_d = 0.1   # 거리 오차 미분 게인 (상대 속도와 연관)
        self.ki_d = 0.01  # 거리 오차 적분 게인
        
        self.dt = dt
        self.integral_error = 0.0
        
    def compute_control(self, spacing_error, rel_vel):
        """
        간격 오차와 상대 속도를 기반으로 Throttle/Brake 값 계산
        """
        # 1. 적분기 업데이트 (Anti-windup 고려)
        self.integral_error += spacing_error * self.dt
        self.integral_error = np.clip(self.integral_error, -10.0, 10.0)
        
        # 2. PID 제어 입력 계산
        # 거리가 멀면(spacing_error > 0) 양수 출력, 속도 차이가 크면(rel_vel > 0) 양수 출력
        desired_accel = (self.kp_d * spacing_error) + (self.ki_d * self.integral_error) + (self.kd_d * rel_vel)
        
        # 3. 가속도를 Throttle과 Brake 행동([-1, 1])으로 매핑
        # 논문의 RL Action Space 매핑(식 9)과 동일한 기준 적용
        action = np.clip(desired_accel, -1.0, 1.0)
        
        throttle = action if action >= 0 else 0.0
        brake = -action if action < 0 else 0.0
        
        return throttle, brake

    def reset(self):
        self.integral_error = 0.0