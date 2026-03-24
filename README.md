# CAV Platoon Safe RL Controller

본 프로젝트는 안전 제약 조건이 포함된 강화학습(Safe Reinforcement Learning)을 이용하여 통신이 불안정한 상황에서도 연결 및 자율주행 차량(CAV) 군집(Platoon)의 종방향 제어를 안전하게 수행하는 PPO-Lag 기반 컨트롤러입니다.

## 🛠 아키텍처 및 환경
- **강화학습 프레임워크**: OmniSafe (PPO-Lag)
- **시뮬레이터**: CARLA (Town04 Highway)
- **미들웨어**: ROS2 (동기화 0.05s Tick)
- **비교군**: Longitudinal PID Controller

## 🚀 설치 방법
git clone https://github.com/tabletfish/Platooning_with_CMDP.git
pip install -r requirements.txt

##터미널2
# (선택) ROS2 환경변수가 기본 세팅이 안 되어 있다면 아래 명령어를 먼저 칩니다.
# source /opt/ros/humble/setup.bash 

cd ~/Platooning_with_CMDP
python carla_bridge.py

#터미널3
cd ~/Platooning_with_CMDP
python main.py
