# motion1 — Motion Planning + RL Hybrid 설계 계획

> 다른 Claude 세션이 이어서 작업할 수 있도록 최대한 self-contained 하게 작성된 plan. 코드 없는 상태에서 시작.

---

## 0. 한 줄 요약

기존 example7/example7_2 에서 박스 적재를 RL 단독으로 학습 시도 → 학습 어려움 (수십 회 reward tuning, 여전히 success 0). 큰 운동 (reach/transport) 은 **motion planning** 으로 즉시 해결, **contact-rich 미세 조정 (grasp / insert) 만 RL** 로 학습하는 hybrid 방식으로 재설계.

---

## 1. 프로젝트 전체 컨텍스트

### 1.1. 시스템

| 항목 | 값 |
|---|---|
| 시뮬레이터 | NVIDIA Isaac Lab / Isaac Sim 4.5 (GPU 병렬) |
| 로봇 | ROBOTIS OMY (6-DOF arm + 4-joint mimic gripper) |
| 로봇 USD | `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd` |
| 로봇 cfg | `source/omy/omy_robot_cfg.py` (`OMY_OFF_SELF_COLLISION_CFG`) |
| RL 라이브러리 | Stable Baselines 3 (PPO / SAC + HER) |
| 환경 frequency | 60 Hz (sim 120 Hz, decimation 2) |
| Episode 길이 | 10초 (= 600 step) |

### 1.2. 로봇 관절 / 그리퍼

- arm joints: `joint1` ~ `joint6` (6 DOF)
- gripper joints: `rh_l1`, `rh_r1_joint`, `rh_l2`, `rh_r2` (4-joint mimic, 동일 명령으로 제어)
- main gripper joint: `rh_l1` (참조용)
- finger bodies: `rh_p12_rn_l2` (left), `rh_p12_rn_r2` (right) — example7 cfg 의 `left_finger_body_name` / `right_finger_body_name`

### 1.3. 박스 / 셀 (예전 example7_2 cfg 와 동일)

**박스**:
- 크기: 0.139 × 0.044 × 0.118 m (가늘고 세워진 형태, 그리퍼는 y 방향 grip)
- 질량: 0.3 kg
- 시작 위치 (월드): (0.45, -0.10, 0.06) ± 2cm randomization

**셀** (3×3 그리드, 그러나 motion1 에서는 1개 셀만 사용해서 검증):
- 셀 내부: 0.17 × 0.06 m
- 격벽: 두께 0.008 m, 높이 0.12 m
- 셀 중심 (월드): (0.25, -0.45, 0.06)
- 박스 → 셀까지 xy 거리 ≈ 0.43 m

### 1.4. 기존 자산 (활용 가능)

| 자산 | 위치 | 비고 |
|---|---|---|
| `example5` 정책 (PPO + LiftEnv) | `checkpoints/example5.zip` + `example5_vecnorm.pkl` | 박스 grasp+lift z=0.20 까지 잘 됨 (frozen) |
| `handoff_states.npz` | `checkpoints/handoff_states.npz` | example5 lift 결과 100개 (z=0.17~0.20, 박스 잡힌 상태). RL 시작 상태로 사용 |
| `example7` 코드 (PlaceEnv) | `source/example7/tasks/place/place_env.py` | reward 함수만 새로 짜면 됨, env 골격 재사용 가능 |
| Grid wall spawn 코드 | `source/example7/tasks/place/place_env.py:_spawn_grid_walls()` | 셀 격벽 USD 생성 코드 |

---

## 2. 왜 motion planning + RL hybrid?

### 2.1. 기존 RL 단독 접근의 한계 (이미 검증된 실패)

- example7 (xy 정렬 + z 유지): v1~v11 까지 reward 튜닝 시도
- 핵심 문제: 정책이 **0.43m 큰 운동 + 1cm 정밀 정렬** 동시 학습 불가
  - 멀리서는 gradient 약하고
  - 가까이 가면 fly-by (통과만 함, 멈춤 X)
  - 박스 한쪽 끝 잡고 z 들면 자연스럽게 기울어짐
- 2~3M step 학습해도 **cumulative_success = 0**

### 2.2. 분해의 정당성

Manipulation 작업의 본질적 분해:
- **Reach / Transport** = "정해진 좌표로 end-effector 이동" → IK + 직선 trajectory 로 1초 안에 해결
- **Grasp** = "박스 표면 접촉 + 그립 닫기, slip 처리" → contact dynamics, RL 강점
- **Insert** = "셀벽과 작은 clearance 안에서 yaw 정렬 후 삽입" → contact-rich, RL 강점

→ RL 학습 task 가 **5cm 이내 미세 조정**으로 좁아짐. 탐험 공간 1/100 → 수렴 10배 빠름 예상.

### 2.3. 각 단계 처리 방식

| 단계 | 처리 | 이유 |
|---|---|---|
| 1. **Reach** (home → 박스 위) | Motion planning | 단순 직선 이동, deterministic |
| 2. **Grasp** (박스 잡기) | **RL** | 박스 slip, 마찰 처리 — contact dynamics |
| 3. **Transport** (z=0.30 까지 lift + 셀 위로 이동) | Motion planning | 단순 이동 |
| 4. **Place + Insert** (yaw 정렬 + 셀 안 삽입) | **RL** | 셀벽 마찰, 작은 clearance, contact-rich |
| 5. **Retract** (셀에서 손 빼고 home) | Motion planning | 단순 이동 |

---

## 3. 폴더 구조 (제안)

```
source/motion1/
├── CLAUDE.md                    # 프로젝트 컨텍스트 (다음 Claude 가 읽을 메모)
├── PLAN.md                      # 이 파일
├── README.md                    # 사용법 (학습/실행 명령)
├── motion/
│   ├── __init__.py
│   ├── ik_solver.py             # IK 솔버 (Isaac Lab Differential IK 활용)
│   ├── trajectory.py            # 직선/곡선 보간
│   └── motion_planner.py        # 통합 (target → trajectory → execute)
├── tasks/
│   ├── __init__.py
│   ├── grasp/
│   │   ├── __init__.py
│   │   ├── grasp_env.py         # Grasp RL env (DirectRLEnv)
│   │   └── grasp_env_cfg.py
│   └── insert/
│       ├── __init__.py
│       ├── insert_env.py        # Insert RL env (DirectRLEnv + GoalEnv)
│       └── insert_env_cfg.py
├── chain/
│   ├── __init__.py
│   └── chain_runner.py          # motion + RL 통합 시퀀스 실행
└── scripts/
    ├── train_grasp.py           # Grasp 학습 (PPO)
    ├── train_insert.py          # Insert 학습 (SAC + HER)
    ├── play_chain.py            # 전체 시퀀스 시연
    ├── collect_pre_grasp.py     # Grasp RL 시작 상태 (motion plan 끝 자세) 수집
    └── collect_pre_insert.py    # Insert RL 시작 상태 (motion plan 끝 자세) 수집
```

---

## 4. 컴포넌트별 상세 spec

### 4.1. Motion Planning 모듈 (`motion/`)

#### `ik_solver.py`

**역할**: end-effector 목표 (xyz + 회전) → 관절 각도

**구현 옵션**:
1. **Isaac Lab 내장 `DifferentialIKController`** 사용 (권장)
   - import: `from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg`
   - Damped least-squares IK, 안정적
2. 직접 구현 (pinocchio 또는 numerical Jacobian) — 권장 X

**API**:
```python
class IKSolver:
    def __init__(self, robot_cfg, finger_offset_local):
        # robot_cfg = OMY_OFF_SELF_COLLISION_CFG
        # finger_offset_local = grip center 가 어느 link 의 어느 좌표인지
        ...

    def solve(self, target_ee_pos_w: torch.Tensor,    # (N, 3) 목표 ee 월드 좌표
              target_ee_quat_w: torch.Tensor = None,  # (N, 4) 옵션
              current_joint_pos: torch.Tensor = None  # (N, num_joints) 시드
             ) -> torch.Tensor:                       # (N, 6) 목표 arm joint
        # 실패 시 raise IKFailure or return None
```

**주의**:
- OMY 의 `body_pos_w` 인덱스에서 `left_finger_body_id` / `right_finger_body_id` 의 평균을 grip center 로 사용
- 또는 wrist link 의 좌표 + offset 으로 정의
- IK target 은 grip center 의 월드 좌표

#### `trajectory.py`

**역할**: 시작 → 목표 사이를 N step trajectory 로 보간

**API**:
```python
def linear_joint_trajectory(joint_start: torch.Tensor,
                            joint_end: torch.Tensor,
                            num_steps: int) -> torch.Tensor:
    # 단순 선형 보간, (num_steps, num_joints) 반환

def cartesian_trajectory_then_ik(ee_start: torch.Tensor,
                                 ee_end: torch.Tensor,
                                 num_steps: int,
                                 ik_solver: IKSolver) -> torch.Tensor:
    # cartesian 직선 보간 → 각 점에서 IK → joint trajectory
    # ee 가 직선으로 움직이게 보장 (joint 보간보다 자연스러움)
```

#### `motion_planner.py`

**API**:
```python
class MotionPlanner:
    def __init__(self, robot, ik_solver):
        self.robot = robot
        self.ik = ik_solver

    def plan_to_pose(self, env_id: int,
                     target_ee_pos_w: torch.Tensor,
                     duration_s: float = 1.0) -> torch.Tensor:
        # 현재 joint 자세 → target ee 위치까지 trajectory
        # duration_s × 60Hz = trajectory 길이

    def execute(self, env_id: int, trajectory: torch.Tensor):
        # trajectory 따라 매 step joint position target 설정
        # 시뮬 step 진행
```

### 4.2. Grasp RL Env (`tasks/grasp/`)

**역할**: 그리퍼가 박스 위 5cm 정도에 도달한 상태에서 시작 → 박스 정확히 잡기

**시작 상태**:
- 박스: (0.45, -0.10, 0.06) ± 2cm
- 그리퍼: 박스 위 5cm (= z ≈ 0.118), 그립 열림
- (motion planner 가 reach 단계 끝낸 자세)

**Observation** (compact, ~20-dim):
- arm joint pos (6)
- arm joint vel (6)
- gripper close state (1)
- ee_pos (월드 → env-relative, 3)
- box_pos (env-relative, 3)
- ee → box 벡터 (3)

**Action**: 7-dim (arm 6 + gripper)

**Reward** (단순):
- `r_approach = exp(-50 * dist(ee, box)²)` — 그립 박스에 접근
- `r_grip_close = (gripper_close > 0.5).float()` — 닫음
- `r_grasp_success = (gripper close AND box height > spawn_z + 1cm).float()` — 박스 살짝 들리면 success
- 가중치 예: 5 / 3 / 50

**종료**:
- success: 박스 z > 시작 z + 1cm AND grip closed AND box near grip → terminate (성공)
- fail: 박스 z < -0.05 (떨어짐) → terminate
- truncated: 5초 (= 300 step)

**알고리즘**: SAC (off-policy, sample efficient) 또는 PPO

**예상 학습**: 200~500k step, 5~15분

### 4.3. Insert RL Env (`tasks/insert/`)

**역할**: 박스 잡힌 채 셀 위 5cm 에 위치한 상태에서 시작 → yaw 정렬 + 셀 안 삽입 + release

**시작 상태**:
- 박스: 그리퍼에 잡힌 채 (0.25, -0.45, 0.18) — 셀 격벽 (0.12) 위 6cm
- yaw: 임의 (또는 ±30도 randomization)
- (motion planner 가 transport 단계 끝낸 자세)

**Observation** (HER GoalEnv 형식):
- core (≈25-dim):
  - arm joint pos/vel (12)
  - gripper close (1)
  - ee_pos rel (3)
  - box_pos rel (3)
  - box_vel (3)
  - box_quat (4) 또는 endpoint_pos rel (3)
- achieved_goal: box xy + yaw 측정값 (e.g., box endpoint xy)
- desired_goal: 셀 중심 xy + 셀 방향 (고정)

**Action**: 7-dim

**Reward**:
- `r_xy_keep = exp(-100 * xy_dist²)` — 셀 중심 위 유지
- `r_yaw = exp(-100 * endpoint_diff²)` — yaw 정렬
- `r_descent = where(xy_aligned & yaw_aligned, exp(-30 * (z - 0.06)²), 0)` — 정렬 후 하강
- `r_release = (deep_enough AND grip_open).float()` — 셀 안에서 그립 열기
- `r_success = (in_cell AND on_floor AND yaw_aligned).float()` — 최종 성공

**종료**:
- success: 박스 셀 안 정렬 + 안착 + 안정 → terminate
- fail: 박스 떨어뜨려서 셀 밖 → terminate
- truncated: 5초

**알고리즘**: SAC + HER (sparse goal-based reward 효과)

**예상 학습**: 500k~1M step, 10~20분

### 4.4. Chain Runner (`chain/chain_runner.py`)

**전체 시퀀스**:

```python
class ChainRunner:
    def __init__(self, env, ik_solver, motion_planner,
                 grasp_policy, insert_policy):
        self.env = env  # MultiBoxEnv 또는 단일박스 env
        self.mp = motion_planner
        self.grasp_policy = grasp_policy
        self.insert_policy = insert_policy

    def run_one_box(self, env_id: int, box_idx: int, target_cell_idx: int):
        # 1. Reach: home → 박스 위 5cm
        box_pos = self.env.get_box_pos(env_id, box_idx)
        ee_target_reach = box_pos + [0, 0, 0.05]  # 5cm 위
        traj = self.mp.plan_to_pose(env_id, ee_target_reach, duration_s=1.5)
        self.mp.execute(env_id, traj)

        # 2. Grasp RL
        for step in range(300):  # max 5초
            obs = self.compute_grasp_obs(env_id, box_idx)
            action = self.grasp_policy.predict(obs, deterministic=True)
            self.env.step(action)
            if self.check_grasp_success(env_id, box_idx):
                break

        # 3. Transport: lift z=0.30 → 셀 위 z=0.18
        cell_pos = self.env.get_cell_pos(env_id, target_cell_idx)
        ee_target_transport_high = cell_pos + [0, 0, 0.30 - 0.06]  # 셀 중심 위 z=0.30
        traj = self.mp.plan_to_pose(env_id, ee_target_transport_high, duration_s=2.0)
        self.mp.execute(env_id, traj)

        ee_target_transport_low = cell_pos + [0, 0, 0.18 - 0.06]  # 셀 중심 위 z=0.18 (격벽 위 6cm)
        traj = self.mp.plan_to_pose(env_id, ee_target_transport_low, duration_s=1.0)
        self.mp.execute(env_id, traj)

        # 4. Insert RL
        for step in range(300):
            obs = self.compute_insert_obs(env_id, box_idx, target_cell_idx)
            action = self.insert_policy.predict(obs, deterministic=True)
            self.env.step(action)
            if self.check_insert_success(env_id, box_idx, target_cell_idx):
                break

        # 5. Retract: home 으로
        traj = self.mp.plan_to_pose(env_id, HOME_EE_POS, duration_s=1.5)
        self.mp.execute(env_id, traj)

    def run_all(self):
        for box_idx in range(self.env.num_boxes):
            self.run_one_box(env_id=0, box_idx=box_idx,
                             target_cell_idx=self.env.box_to_cell[box_idx])
```

---

## 5. 구현 순서 (단계별)

> 각 단계마다 즉시 검증 (시각화 또는 메트릭).

### Step 1 — Motion Planning 검증 (1~2시간)

1. `motion/ik_solver.py` 작성 (Isaac Lab `DifferentialIKController` 래핑)
2. `motion/trajectory.py` 작성 (선형 보간)
3. `scripts/test_motion.py` 작성:
   - 단일박스 env (example7 의 PlaceEnv 빌려서 사용)
   - home → 박스 위 5cm → 셀 위 z=0.30 → 셀 위 z=0.18 → home 시퀀스
   - 시각화로 부드럽게 움직이는지 확인
   - 박스가 셀벽에 충돌 안 하는지 확인 (z=0.18 = 격벽 0.12 위 6cm)

**검증 기준**: 시뮬 윈도우에서 사람이 봤을 때 자연스러운 reach/transport/retract 움직임.

**위험 요소**: IK singularity (관절 한계 근처에서 IK 실패). Solution: 시드 자세를 잘 설정 + damping 강하게.

### Step 2 — Grasp RL Env (30분 ~ 1시간)

1. `tasks/grasp/grasp_env.py` 작성 (DirectRLEnv 상속, example7 골격 참고)
2. 시작 상태: motion planner 가 reach 끝낸 자세 (그립이 박스 위 5cm)
3. Reset 함수: 박스 위치 randomize ± 2cm, 그립 위치 reset (motion plan 안 거치고 바로 그 자세로)
4. `scripts/train_grasp.py` 작성 (SAC, num_envs=64)
5. 200~500k step 학습
6. `scripts/play_grasp.py` 로 시각화

**검증 기준**: 박스를 안정적으로 잡고 1cm 살짝 들기 성공률 ≥ 90%.

### Step 3 — Insert RL Env (30분 ~ 1시간)

1. `tasks/insert/insert_env.py` 작성 (DirectRLEnv + GoalEnv 형식, example7_2 참고)
2. 시작 상태: 박스 그리퍼에 잡힌 채 (0.25, -0.45, 0.18) — 셀 위 6cm
3. Yaw randomize ±30도, xy 위치 ±2cm
4. `scripts/train_insert.py` 작성 (SAC + HER)
5. 500k~1M step 학습
6. 시각화 검증

**검증 기준**: 셀 안에 박스 정렬 + 삽입 + release 성공률 ≥ 70%.

### Step 4 — Chain 통합 (1~2시간)

1. `chain/chain_runner.py` 작성
2. `scripts/play_chain.py`: 1박스 단일 시퀀스 (Reach → Grasp → Transport → Insert → Retract)
3. 단일박스 검증
4. 3박스 + 3×3 grid (또는 1×3) 확장

**검증 기준**: 3 박스 적재 성공률 ≥ 70%.

---

## 6. 기존 자산 재사용

| 자산 | 어떻게 재사용 |
|---|---|
| `example5.zip` | **사용 안 함** (motion planning 으로 reach 대체) |
| `handoff_states.npz` | **사용 안 함** (motion planning 이 시작 자세 결정) |
| `example7/tasks/place/place_env.py` | env 골격 (DirectRLEnv 상속, _setup_scene 등) 참고만. reward 와 reset 은 새로 |
| `OMY_OFF_SELF_COLLISION_CFG` | 그대로 사용 |
| `_spawn_grid_walls()` | 그대로 사용 |
| `place_env_cfg.py` 의 grid 파라미터 (cell_inner_x 등) | 그대로 사용 |

**완전히 새로 짤 부분**:
- `motion/` 전체
- `grasp_env.py`, `insert_env.py` (env 자체는 새로지만 기존 env 코드 참고하면 1시간 안에 가능)
- `chain_runner.py`
- 학습 스크립트

---

## 7. 학습 cfg 권장값

### Grasp (SAC)
- `num_envs`: 64
- `timesteps`: 500_000 (충분, 짧은 task)
- `learning_starts`: 5000
- `gradient_steps`: 1
- `batch_size`: 256
- `gamma`: 0.99
- `tau`: 0.005
- `episode_length_s`: 5.0

### Insert (SAC + HER)
- `num_envs`: 64
- `timesteps`: 1_000_000 (HER 라 더 길게)
- `replay_buffer_class`: HerReplayBuffer
- `replay_buffer_kwargs`: `{"goal_selection_strategy": "future", "n_sampled_goal": 4}`
- `learning_starts`: 10000
- `episode_length_s`: 5.0

---

## 8. 주의사항 / 함정

1. **OMY USD 절대경로 하드코딩**:
   - `omy_robot_cfg.py` 에 USD 경로가 절대 경로로 박혀있음
   - `/home/jaewoo/IsaacLab/source/omy_f3m_urdf/OMY.usd`
   - 다른 환경에서는 수정 필요

2. **Gripper mimic**:
   - 4-joint 가 같은 명령으로 움직여야 함 (`_actions_to_dof()` 참고)
   - 예전 example5 / example7 코드에 mimic 처리 있음 — 그대로 가져와야 함

3. **IK 시드 자세 중요**:
   - 매 step IK 풀 때 현재 joint pose 시드로 사용
   - Singularity 근처면 IK 발산 가능 → damping 키우거나 fallback

4. **Motion planner 시 박스 떨어뜨림 방지**:
   - Transport 시 박스를 잡고 있는데, 정책 없이 (motion plan 만 으로) 박스가 잡힌 채로 유지되어야 함
   - Gripper joint 는 명령을 닫힌 상태로 유지 (action[6] = max_close)
   - PhysX 에서 그립 마찰력 충분한지 검증 필요

5. **Insert RL 시작 자세 randomization**:
   - Yaw ±30도 정도 randomize 해야 학습 generalize
   - 너무 좁게 randomize 하면 정책이 좁은 분포만 학습 → motion plan 출력에 sensitive

6. **Reward log 형식**:
   - 기존 train.py 의 callback 이 `env.reward_log` 딕셔너리를 100k step 마다 print
   - 새 env 도 같은 패턴으로 작성 권장 (디버깅 쉬움)

7. **Sim2real 고려**:
   - 이 단계에서는 ground truth pose 사용 (간단)
   - 나중 vision 통합 시 grasp/insert RL 만 vision obs 로 재학습
   - Motion planning 부분은 vision 영향 X (좌표만 알면 됨)

---

## 9. 다음 작업자 (다른 Claude) 가 시작할 때

1. 이 PLAN.md 와 함께 다음 파일들 읽기:
   - `source/example7/tasks/place/place_env.py` (env 구조 참고)
   - `source/example7/tasks/place/place_env_cfg.py` (cfg 파라미터)
   - `source/example7/scripts/train.py` (학습 스크립트 패턴)
   - `source/example5/tasks/lift/lift_env.py` (\_actions_to_dof, gripper mimic 처리 참고)
   - `source/omy/omy_robot_cfg.py` (로봇 cfg)

2. 사용자 환경 정보:
   - Conda env: `env_isaaclab` (`/home/jaewoo/miniconda3/envs/env_isaaclab/`)
   - 학습 실행: `./isaaclab.sh -p source/<path>/scripts/train.py --headless --num_envs 64`
   - 시각화: `./isaaclab.sh -p source/<path>/scripts/play.py --checkpoint ...`

3. 권장 진행:
   - Step 1 (motion planning) 먼저 끝까지 검증 → 시각화로 자연스러운 움직임 확인
   - 그 후 Step 2 (Grasp RL) 진행
   - Step 3 (Insert RL) 진행
   - Step 4 (Chain) 마지막

4. 사용자 스타일:
   - 한국어로 설명 선호
   - 코드 변경 후 즉시 시각화로 검증 원함
   - 페널티 (음수 reward) 최대한 피함, all-positive reward 선호
   - 가중치 표준 RL 스케일 (1~100 범위) 선호 (큰 값 100~1000 회피)
   - 학습 너무 길게 잡지 말 것 — 200k~2M step 범위에서 결정

5. 컨텍스트 보존:
   - 자율 메모리 시스템 사용 중 (`/home/jaewoo/.claude/projects/-home-jaewoo-IsaacLab/memory/`)
   - `MEMORY.md` 와 관련 메모 읽으면 사용자 선호 / 진행 상황 파악 가능

6. 절대 건드리지 말 것:
   - `source/example5/` (frozen, 작동 검증됨)
   - `checkpoints/example5.zip`, `example5_vecnorm.pkl`, `handoff_states.npz`

---

## 10. 외부 자문용 요약 (있으면 좋음)

기존 RL 단독 시도 실패 → motion planning + RL 분해. 외부 RL 전문가에게 의견 받을 만한 부분:

- IK 솔버 선택 (Differential IK vs analytical IK)
- Insert RL 의 reward 설계 (특히 contact-rich 부분)
- Sim2real 시 motion planning 의 robustness 검증 방법
- Multi-stage hybrid (BC + RL, residual policy 등) 가능성

기존 `source/example7/HANDOFF_SUMMARY.md` 가 외부 자문용 문서. motion1 진행 후 업데이트 권장.

---

## 끝

질문 있으면 사용자에게 직접 물어보고 진행하세요. PLAN 자체는 사용자가 confirm 한 상태이므로 이대로 시작 OK.
