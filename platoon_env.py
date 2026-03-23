import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random

class PlatoonSafeEnv(gym.Env):
    def __init__(self):
        super(PlatoonSafeEnv, self).__init__()
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(7,), dtype=np.float32)
        
        # --- 논문 Table I 파라미터 초기화 ---
        self.dt = 0.05
        self.p_success = 0.9
        
        # CTG 정책 파라미터 [cite: 183]
        self.h = 1.0         # Time headway (s)
        self.d_0 = 7.0       # Minimum standstill distance (m)
        
        # 보상 가중치 파라미터 [cite: 183]
        self.omega_1 = 0.1
        self.omega_2 = 0.25
        self.omega_3 = 0.25
        self.Q = np.array([[0.1, 0.0], [0.0, 0.1]]) # Q 매트릭스 (가정값)
        
        # 페널티 경계값 및 스케일링 계수 [cite: 183]
        self.d_far = 8.0
        self.d_close = -4.0
        self.alpha_1 = 20.0
        self.alpha_2 = 10.0
        
        # 안전 THW 임계값 [cite: 183]
        self.tau_safe = 1.2
        self.tau_danger = 1.0
        # ------------------------------------
        
        self.last_comm_data = {'spacing_error': 0.0, 'rel_vel': 0.0, 'prec_accel': 0.0}
        self.time_since_last_comm = 0.0
        self.prev_accel = 0.0  # Jerk 계산용 이전 가속도 저장

    def step(self, action):
        a_i = action[0]
        throttle = a_i if a_i >= 0 else 0.0
        brake = -a_i if a_i < 0 else 0.0
        
        # CARLA 제어 명령 전송 및 0.05초 대기 로직 (생략)
        
        comm_success = random.random() <= self.p_success
        obs = self._get_observation(comm_success)
        
        # 현재 상태 변수 추출
        delta_d = obs[0]  # Spacing error
        delta_v = obs[1]  # Relative velocity
        v_i = obs[3]      # Ego velocity
        
        # --- 보상 및 비용 계산 ---
        reward = self._compute_reward(delta_d, delta_v, a_i, self.prev_accel)
        cost = self._compute_cost(delta_d, v_i)
        
        # 다음 스텝을 위해 현재 가속도 업데이트
        self.prev_accel = a_i 
        
        terminated = False
        truncated = False
        info = {'cost': cost}
        
        return obs, reward, terminated, truncated, info

    def _get_observation(self, comm_success):
        # 관측값 로직 (이전 스텝과 동일하므로 생략)
        return np.zeros(7, dtype=np.float32)

    def _compute_reward(self, delta_d, delta_v, a_i, prev_a_i):
        # 1. 제어 효율성 비용 (Control efficiency cost) 
        state_vec = np.array([delta_d, delta_v])
        r_cont = state_vec.T @ self.Q @ state_vec
        
        # 2. 트래픽 방해 페널티 (Traffic disturbance penalty) [cite: 166]
        r_traf = a_i ** 2
        
        # 3. 승차감/저크 페널티 (Jerk-related penalty) 
        r_jerk = (a_i - prev_a_i) ** 2
        
        # 4. 허용 범위 이탈 페널티 (Spacing deviation penalty) [cite: 172]
        if delta_d > self.d_far:
            r_penalty = ((delta_d - self.d_far) ** 2) / self.alpha_1
        elif delta_d < self.d_close:
            r_penalty = ((delta_d - self.d_close) ** 2) / self.alpha_2
        else:
            r_penalty = 0.0
            
        # 5. 최종 보상 산출 [cite: 201]
        reward = np.exp(-(self.omega_1 * r_cont + self.omega_2 * r_traf + self.omega_3 * r_jerk)) - r_penalty
        return float(reward)

    def _compute_cost(self, delta_d, v_i):
        # 1. CTG 기반 목표 거리 및 실제 거리 계산 
        d_ctg = v_i * self.h + self.d_0
        d_i = delta_d + d_ctg  # 실제 거리 d_i(t)
        
        # 2. THW 계산 (0으로 나누기 방지)
        thw = d_i / v_i if v_i > 0.1 else float('inf')
        
        # 3. 3단계 위험 모델에 따른 비용 산출 [cite: 204, 205, 206, 207, 208, 209, 210, 211, 212]
        if thw >= self.tau_safe:
            cost = 0.0
        elif thw <= self.tau_danger:
            cost = 1.0
        else:
            cost = (self.tau_safe - thw) / (self.tau_safe - self.tau_danger)
            
        return float(cost)