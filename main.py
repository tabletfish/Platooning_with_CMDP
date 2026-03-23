import rclpy
import gymnasium as gym
from gymnasium.envs.registration import register
import omnisafe
import torch

# 앞서 구현한 환경 클래스 임포트 (파일 이름이 platoon_env.py일 경우)
# from platoon_env import PlatoonSafeEnv 

def main():
    # 1. ROS2 미들웨어 초기화
    # CARLA와 통신할 노드를 생성하기 위해 가장 먼저 실행되어야 합니다.
    rclpy.init()

    # 2. Custom Environment Gymnasium 레지스트리 등록
    # OmniSafe가 문자열 ID('PlatoonSafe-v0')로 환경을 찾을 수 있도록 등록합니다.
    register(
        id='PlatoonSafe-v0',
        entry_point='platoon_env:PlatoonSafeEnv', # '모듈명:클래스명' 구조
        max_episode_steps=3000, # 에피소드 최대 길이 (논문의 step 수 참고)
    )

    # 3. OmniSafe(PPO-Lag) 하이퍼파라미터 및 환경 설정
    # 논문의 Table I과 PPO-Lag 알고리즘 특성에 맞춘 설정값입니다.
    custom_cfgs = {
        'train_cfgs': {
            'total_steps': 1000000,       # 전체 학습 스텝 수 (수렴 시까지)
            'vector_env_nums': 1,         # 단일 환경 사용 (ROS2 동기화 문제 방지)
            'torch_threads': 4,
        },
        'algo_cfgs': {
            'update_cycle': 2000,         # PPO 업데이트 주기
            'update_iters': 10,           # 에포크 당 업데이트 횟수
            'cost_limit': 5.0,            # 논문 Table I: 누적 안전 비용 제약 조건 (d = 5)
            'penalty_coef': 0.0,          # 초기 라그랑주 승수 (lambda >= 0)
            'use_cost': True,             # 안전 제약 조건 사용 활성화
        },
        'logger_cfgs': {
            'use_wandb': False,           # 필요시 True로 변경하여 Weights & Biases 연동
            'save_model_freq': 10,        # 모델 저장 빈도 (에포크 기준)
            'log_dir': './runs/platoon_ppo_lag'
        }
    }

    print("======================================================")
    print("🚀 플래툰 종방향 제어를 위한 PPO-Lag 학습을 시작합니다.")
    print("통신 실패 확률(p_success=0.9) 및 THW 제약 조건이 적용됩니다.")
    print("======================================================")

    # 4. 에이전트 초기화 (PPO-Lag 알고리즘 적용)
    env_id = 'PlatoonSafe-v0'
    agent = omnisafe.Agent('PPOLag', env_id, custom_cfgs=custom_cfgs)

    # 5. 정책 학습 실행 (이 과정에서 환경의 reset()과 step()이 반복 호출됨)
    try:
        agent.learn()
    except KeyboardInterrupt:
        print("\n🛑 사용자에 의해 학습이 중단되었습니다.")
    finally:
        # 6. 학습 종료 후 ROS2 자원 정리
        rclpy.shutdown()
        print("✅ ROS2 미들웨어가 안전하게 종료되었습니다.")

if __name__ == '__main__':
    main()