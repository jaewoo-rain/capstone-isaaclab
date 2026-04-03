# My Grasp RL

Isaac Lab Direct workflow 기반 Franka grasp + lift task.

## 목적
- 랜덤 위치의 큐브를 집고
- 일정 높이까지 들어올리는 정책 학습

## 환경
- Robot: Franka Panda (Isaac Lab 기본 제공)
- Object: cube
- Action: joint target delta + gripper command
- Observation: joint states, ee pose, object pose, relative vector
- Reward: reach + align + grasp + lift

## 실행 예시

### 1) Isaac Lab root에서 PYTHONPATH 추가
```bash
export PYTHONPATH=/home/jaewoo/IsaacLab/source:$PYTHONPATH

# grasp
## 학습
./isaaclab.sh -p /path/to/my_grasp_rl/source/my_grasp_rl/scripts/train_sb3.py --headless
### 새로 학습
./isaaclab.sh -p source/my_grasp_rl/scripts/train_sb3.py
### 이어서 학습
./isaaclab.sh -p source/my_grasp_rl/scripts/train_sb3.py --resume
### 100만 step 더 하기
./isaaclab.sh -p source/my_grasp_rl/scripts/train_sb3.py --resume --timesteps 1000000
## 재생
./isaaclab.sh -p /path/to/my_grasp_rl/source/my_grasp_rl/scripts/play_sb3.py --num_envs 1

# stacking
## 학습
### 새로 학습
./isaaclab.sh -p source/my_grasp_rl/scripts/train_stacking_sb3.py --timesteps 1000000
### 이어서 학습
./isaaclab.sh -p source/my_grasp_rl/scripts/train_stacking_sb3.py --resume --timesteps 1000000
### 개수 조절 학습
./isaaclab.sh -p source/my_grasp_rl/scripts/train_stacking_sb3.py --num_envs 32 --timesteps 300000
## 테스트
./isaaclab.sh -p source/my_grasp_rl/scripts/play_stacking_sb3.py --num_envs 1

# good-robot
./isaaclab.sh -p source/my_grasp_rl/scripts/train_good_robot_sb3.py
./isaaclab.sh -p source/my_grasp_rl/scripts/train_good_robot_sb3.py --resume --timesteps 1000000
./isaaclab.sh -p source/my_grasp_rl/scripts/train_good_robot_sb3.py --num_envs 1
