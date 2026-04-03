### 1) Isaac Lab root에서 PYTHONPATH 추가
```bash
export PYTHONPATH=/home/jaewoo/IsaacLab/source:$PYTHONPATH
```

# 실행
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task stacking --resume --timesteps 1000000
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task good --resume --timesteps 1000000
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task grasp --resume --headless --timesteps 1000000

# 테스트
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task stacking --num_envs 1
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task good --num_envs 1
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task grasp --num_envs 1
