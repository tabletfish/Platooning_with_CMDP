import numpy as np

class LeaderTrajectoryGenerator:
    def __init__(self, dt=0.05):
        self.dt = dt
        self.time = 0.0
        # 평가 시나리오에 사용될 타겟 속도 (m/s) [cite: 220]
        self.target_speeds = [15.0, 17.5, 20.0]
        self.current_target_speed = np.random.choice(self.target_speeds)
        self.current_velocity = 0.0

    def get_next_velocity(self):
        """
        시간 흐름에 따른 리더 차량의 목표 속도 프로파일 생성 (Unseen profile 모사) 
        """
        self.time += self.dt
        
        # 0 ~ 40초: 무작위 타겟 속도로 가속 및 순항 [cite: 220]
        if self.time < 40.0:
            if self.current_velocity < self.current_target_speed:
                self.current_velocity += 1.0 * self.dt  # 1.0 m/s^2 가속
            else:
                self.current_velocity = self.current_target_speed
                
        # 40 ~ 45초: 안전이 위협받는 급감속 상황 연출 (Emergency braking) [cite: 221]
        elif 40.0 <= self.time < 45.0:
            self.current_velocity -= 4.0 * self.dt # -4.0 m/s^2 급감속
            self.current_velocity = max(0.0, self.current_velocity)
            
        # 45초 이후: 다시 가속하여 순항
        else:
            if self.current_velocity < 15.0:
                self.current_velocity += 1.5 * self.dt
                
        return self.current_velocity

    def reset(self):
        self.time = 0.0
        self.current_velocity = 0.0
        self.current_target_speed = np.random.choice(self.target_speeds)