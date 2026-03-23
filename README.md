# CAV Platoon Safe RL Controller

본 프로젝트는 안전 제약 조건이 포함된 강화학습(Safe Reinforcement Learning)을 이용하여 통신이 불안정한 상황에서도 연결 및 자율주행 차량(CAV) 군집(Platoon)의 종방향 제어를 안전하게 수행하는 PPO-Lag 기반 컨트롤러입니다.

## 🛠 아키텍처 및 환경
- **강화학습 프레임워크**: OmniSafe (PPO-Lag)
- **시뮬레이터**: CARLA (Town04 Highway)
- **미들웨어**: ROS2 (동기화 0.05s Tick)
- **비교군**: Longitudinal PID Controller

## 🚀 설치 방법
```bash
pip install -r requirements.txt
# (주의: ROS2 Humble 또는 Foxy 설치 및 source 환경 설정이 사전 요구됩니다.)