# rl/ — grasp RL 정책 sim2real 실행

motion1 에서 학습한 **grasp 정책**(`checkpoints/motion1_grasp.zip`)을 실제 OMY-F3M
로봇에서 실행한다. 박스 좌표(`x, y, yaw`)를 받아 EE 를 박스 위로 정렬한 뒤
하강·파지·들기까지 수행한다.

> 좌표는 **지금은 수동 입력**. 나중에 카메라가 같은 `(x, y, yaw)` 자리를 채운다.

---

## 핵심 아이디어 — 왜 "내부 롤아웃"인가

grasp 정책은 sim 에서 **60Hz 폐루프**(매 step 작은 Δxy/Δyaw)로 학습됐다.
하지만 실제 로봇 인프라(MoveIt plan 5s + 실행 6s, blocking)는 60Hz 폐루프를
**못 돌린다**(최대 ~0.09Hz).

→ 그래서 정책을 **파이썬 안에서 가상으로** 굴린다. 박스는 고정이고 EE 만 sim 의
action 적용식과 똑같이 적분하면, sim 폐루프 정렬과 거의 동일한 **최종 정렬
EE pose(xy, yaw)** 가 나온다. 그 한 점만 실제 로봇으로 이동한다.

```
박스 (x,y,yaw) ──┐
현재 EE (TF2) ───┼─▶ [grasp_rollout: 정책 60Hz 가상 롤아웃] ─▶ 최종 정렬 EE (xy,yaw)
                 │                                                    │
                 └────────────────────────────────────────────────── ▼
                              [MoveIt 단일 이동] pre_grasp → grasp → close → lift
```

---

## 파일

| 파일 | 역할 |
|------|------|
| `config.py` | 모든 상수 (sim env 에서 그대로 — **변경 금지**). action scale, 정렬 임계값, z 높이, 학습 분포, grip offset |
| `grasp_policy.py` | SB3 PPO + VecNormalize 로더 (IsaacLab 무의존). `obs(6) → action(3)` |
| `grasp_rollout.py` | 내부 롤아웃. `(현재 EE, 박스) → 최종 정렬 EE pose` |
| `run_grasp.py` | 메인. 박스 좌표 수동 입력 → 롤아웃 → MoveIt 이동 → 하강/파지/들기 |
| `checkpoints/` | `motion1_grasp.zip` + `_vecnorm.pkl` (motion1 에서 복사, 재사용) |

재사용하는 외부 코드: `../run_pick_place.py`(plan/실행/그리퍼 헬퍼),
`../../real_moveit_common.py`(MoveIt goal, TF2, quat).

---

## 의존성 / 실행 환경

- 추론: `stable-baselines3 2.7.1`, `numpy`, `torch` (conda `env_isaaclab` 에 이미 있음). **IsaacLab/Isaac Sim 불필요.**
- 로봇 통신: `rclpy` + MoveIt + OMY 컨트롤러 (별도 bringup)

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
```

사전 조건 (별도 터미널에서 로봇 bringup):
```bash
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

---

## ⚠️ sim2real 검증 순서 (반드시 이 순서로)

자세 매핑에 **미검증 리스크**가 있다. sim 의 EE 기준자세 `(0,1,0,0)` 와 실제 로봇
수직파지 `[0.5063,0.4936,0.4936,0.5063]` 가 숫자상 다르다(top-down vs side-grasp 가능성).
yaw 회전축(world Z)은 같다고 분석됐지만 실물로 확인 전엔 단정 불가. 그래서:

**1단계 — dry-run 으로 계획만 확인**
```bash
cd /home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/rl
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0
```
롤아웃 수렴(`status=converged`)과 각 step plan `ok=True` 를 확인.

**2단계 — yaw 정렬 끄고 자세부터 (실제 실행)**
```bash
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0 \
    --no-align-yaw --execute --confirm EXECUTE_GRASP
```
EE 가 박스 위 수직파지 자세로 내려가 잡는지 눈으로 확인. 자세가 틀리면 여기서 멈추고
`config.GRIP_CENTER_OFFSET_XY` / orientation 매핑을 조정.

**3단계 — yaw 정렬 활성화 (전체)**
```bash
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3 \
    --execute --confirm EXECUTE_GRASP
```
박스를 회전시켜 놓고 EE 가 yaw 까지 맞춰 잡는지 확인.

---

## 안전장치

- **기본 dry-run** — `--execute --confirm EXECUTE_GRASP` 둘 다 있어야 로봇이 움직임
- **단계별 타이핑 확인** — 각 step 이름을 직접 입력해야 진행 (`--no-step-prompts` 로 생략)
- **3단계 guard** (run_pick_place 와 동일) — workspace / MoveIt plan delta / actual delta
- **롤아웃 미수렴 시 실행 거부** — 박스 좌표/시작 자세가 학습 분포 밖이면 멈춤
- **박스 분포 경고** — 학습 범위(`x≈0.45±0.10, y≈-0.10±0.10, yaw ±80°`) 밖이면 경고, 실행 차단
- 경고를 무시하려면 `--force` (주의)

---

## 좌표 / 주의사항

- 박스 좌표는 **base(link0) 기준** `x[m], y[m], yaw[rad]`. `x`=앞쪽, `y`=오른쪽(음수).
- **시작 EE 를 박스 근처(30cm 이내)**, 수직파지 자세로 두고 시작할 것. 너무 멀면 정책이
  학습하지 않은 영역이라 롤아웃이 발산한다.
- z 높이(approach/grasp)는 정책과 무관. YAML `waypoints['1']/['2']` 에서 읽거나
  `--approach-z/--grasp-z` 로 지정.
- `config.GRIP_CENTER_OFFSET_XY`: sim 정책은 grip center(양 finger 평균)를 정렬하는데
  실제 TF 는 link6 를 읽는다. 둘 사이 offset 이 있으면 채울 것 (1차값 0).

---

## 단위 검증 (로봇 없이)

```bash
python3 grasp_policy.py     # 정책 로드 + dummy 추론
python3 grasp_rollout.py    # 박스로 수렴하는지 (status=converged 기대)
```
