SB3_PPO_CFG = {
    "policy": "MlpPolicy", # 일반적인 fully connected network
    "n_steps": 64, # 한 번에 모을 데이터 길이
    "batch_size": 16384, # 학습할 때 사용하는 데이터 묶음 크기, batch_size ≤ n_steps * num_envs
    "learning_rate": 3e-4,
    "gamma": 0.99, # 미래 보상 할인율, 1에 가까울수록 long-term 고려
    "gae_lambda": 0.95, # advantage 계산 방식, 1에 가까울수록 bias↓ variance↑
    "clip_range": 0.2, # 정책 업데이트를 제한하는 범위(20%까지만 정책을 바꿀수 있음)
    "ent_coef": 0.0, # 탐험 유도 정도 (초기에 너무 못찾을때 올리기)
    "vf_coef": 0.5, # value loss 가중치, value 와 policy에서 value의 비율을 얼마나 둘건지
    "max_grad_norm": 1.0, # gradient 폭주 방지
    "verbose": 1,
    # "tensorboard_log": "./logs/sb3/grasp_franka", # 학습 로그 저장 위치
}