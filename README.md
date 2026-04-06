# CAV Platoon Safe RL Controller

본 프로젝트는 CARLA + ROS2 + OmniSafe 기반으로, 통신 열화 환경에서 CAV platoon 종방향 제어를 Safe RL로 학습하는 실험 저장소입니다.

## 핵심 구성
- 강화학습 프레임워크: OmniSafe (`PPOLag`)
- 환경 클래스: `platoon_env.py` (`CMDP` 기반)
- ROS2 노드: `ros2_node.py`
- CARLA bridge: `carlar_bridge.py`
- 학습 진입점: `main.py`

## 현재 확인된 실행 경로
1. mock smoke test
2. ROS live observation smoke test
3. ROS live short train smoke test

## smoke test 스크립트
```bash
cd /home/jungjinwoo/Platooning_with_CMDP
./scripts/smoke_mock_train.sh
./scripts/smoke_live_step.sh
./scripts/smoke_live_train.sh
```

## live 실행 순서
터미널 1:
```bash
cd /home/jungjinwoo/Platooning_with_CMDP
./scripts/start_carla_ros2.sh
```

터미널 2:
```bash
cd /home/jungjinwoo/Platooning_with_CMDP
./scripts/start_bridge.sh
```

터미널 3:
```bash
cd /home/jungjinwoo/Platooning_with_CMDP
./scripts/train_live_full.sh
```

## 주요 환경 변수
- `PLATOON_USE_ROS`: `1`이면 ROS live 경로 사용
- `PLATOON_TOTAL_STEPS`: 전체 학습 step 수
- `PLATOON_STEPS_PER_EPOCH`: epoch 당 rollout step 수
- `PLATOON_UPDATE_ITERS`: update 반복 횟수
- `PLATOON_MAX_EPISODE_STEPS`: episode 길이

## 현재 상태
- `main.py`, `platoon_env.py`, `ros2_node.py`, `carlar_bridge.py` 기준으로 mock/live smoke test는 모두 통과함
- live 경로에서는 CARLA 기반 상태값이 observation으로 들어오도록 연결됨
- reward/cost는 아직 논문 식 기준으로 추가 정리가 필요함
