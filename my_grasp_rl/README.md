### 1) Isaac Lab root에서 PYTHONPATH 추가
```bash
export PYTHONPATH=/home/jaewoo/IsaacLab/source:$PYTHONPATH
```

# 실행
--task : 어떤 모델로 돌릴지
--num_envs : 몇개의 로봇 돌릴지
--name : 파일 이름
--headless : UI없이
--no : 저장 없이
--resume : 저장된 모델 볼러와서 사용
--timesteps : step 몇번 돌릴지
---
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task stacking --timesteps 1000000 --name test1
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task good --resume --timesteps 1000000 --no 
./isaaclab.sh -p source/my_grasp_rl/scripts/train.py --task grasp --resume --headless --timesteps 1000000

# 테스트
--task : 어떤 모델로 돌릴지
--num_envs : 몇개의 로봇 돌릴지
--checkpoint : 위치+파일 이름
--name : 파일 이름
--headless : UI없이
---
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task stacking --num_envs 1
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task good --num_envs 1
./isaaclab.sh -p source/my_grasp_rl/scripts/play.py --task grasp --num_envs 1

# URDF
saacsim/extscache/isaacsim.asset.importer.urdf-2.3.10+106.4.0.lx64.r.cp310/data/urdf/robots/franka_description/robots/panda_arm_hand.urdf