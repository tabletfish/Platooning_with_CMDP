import rclpy
import torch
import numpy as np
import omnisafe
from omnisafe.evaluator import Evaluator

# 다중 차량 제어를 위해 기존 ROS2 노드를 확장한 클래스가 필요합니다.
# (가칭: MultiPlatoonROS2Node)

def evaluate_platoon(model_dir):
    rclpy.init()
    
    # 1. 평가용 다중 차량 ROS2 노드 초기화
    # Follower 1, 2, 3의 상태를 모두 구독하고 제어 명령을 각각 발행해야 합니다.
    # ros2_node = MultiPlatoonROS2Node(num_followers=3)
    
    print("======================================================")
    print("📊 학습된 PPO-Lag 모델을 로드하여 다중 차량 평가를 시작합니다.")
    print(f"모델 경로: {model_dir}")
    print("======================================================")

    # 2. OmniSafe Evaluator를 이용해 학습된 정책(Policy) 로드
    # render_mode를 설정하여 CARLA 시뮬레이션 화면과 연동할 수 있습니다.
    evaluator = Evaluator(render_mode='human')
    evaluator.load_saved(
        save_dir=model_dir,
        model_name='epoch-1000.pt' # 실제 저장된 모델 이름으로 변경
    )
    
    # 평가 환경 파라미터 
    num_followers = 3
    num_episodes = 100 # 논문 기준 평가 반복 횟수 [cite: 224]
    max_steps = 3000
    
    # 평가 지표 기록용 딕셔너리
    metrics = {
        'control_efficiency': np.zeros(num_followers),
        'traffic_disturbance': np.zeros(num_followers),
        'jerk_cost': np.zeros(num_followers),
        'thw_cost': np.zeros(num_followers),
        'caution_duration': np.zeros(num_followers),
        'danger_duration': np.zeros(num_followers)
    }

    # 3. 평가 루프 시작 (논문의 Unseen 리더 속도 프로파일 시나리오 적용)
    for episode in range(num_episodes):
        # ros2_node.reset_simulation(scenario='unseen_profile')
        
        # 각 차량의 이전 가속도 저장용 (Jerk 계산)
        prev_accels = np.zeros(num_followers)
        
        for step in range(max_steps):
            # 3.1. 각 차량의 현재 상태 관측 (통신 성공 확률 p_success 반영)
            # states = ros2_node.get_all_follower_states()
            states = np.zeros((num_followers, 7)) # 임시 더미 데이터
            
            actions = np.zeros(num_followers)
            
            # 3.2. 단일 학습 정책(Policy)을 모든 차량에 배포(공유)하여 행동 추론
            for i in range(num_followers):
                obs_tensor = torch.as_tensor(states[i], dtype=torch.float32)
                # 평가 시에는 탐험(Exploration) 노이즈 없이 결정론적(Deterministic) 행동 추출
                action_tensor = evaluator.actor.predict(obs_tensor, deterministic=True)
                actions[i] = action_tensor.item()
            
            # 3.3. 행동(a_i)을 Throttle/Brake로 변환 후 CARLA로 전송
            throttles = np.where(actions >= 0, actions, 0.0)
            brakes = np.where(actions < 0, -actions, 0.0)
            # ros2_node.publish_all_controls(throttles, brakes)
            
            # 3.4. 0.05초 틱 진행 및 동기화 대기
            # ros2_node.tick_and_wait(0.05)
            
            # 3.5. 지표(Metric) 계산 및 누적 (논문 Fig. 5 기준) 
            for i in range(num_followers):
                a_i = actions[i]
                prev_a_i = prev_accels[i]
                # THW, Control cost 등 계산 (환경 클래스의 계산식 재사용)
                # metrics['traffic_disturbance'][i] += a_i ** 2
                # metrics['jerk_cost'][i] += (a_i - prev_a_i) ** 2
                # (THW가 danger 구간이면 danger_duration += 1 등)
                
            prev_accels = actions.copy()
            
    # 4. 결과 평균 계산 및 출력
    print("\n✅ 평가 완료! 평균 지표 결과:")
    for i in range(num_followers):
        print(f"--- Follower {i+1} ---")
        print(f"Control Efficiency Cost: {metrics['control_efficiency'][i] / (num_episodes * max_steps):.4f}")
        print(f"Danger Region Duration: {metrics['danger_duration'][i] / num_episodes:.1f} steps")

    rclpy.shutdown()

if __name__ == '__main__':
    # 학습된 모델이 저장된 디렉토리 경로 지정
    model_directory = './runs/platoon_ppo_lag/PPO-Lag-{PlatoonSafe-v0}' 
    evaluate_platoon(model_directory)