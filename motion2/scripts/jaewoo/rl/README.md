

x좌표, y좌표 지정해서 실행
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3 \
    --execute --confirm EXECUTE_GRASP











# rl/ — grasp RL 정책 sim2real 실행

motion1 에서 학습한 **grasp 정책**(`checkpoints/motion1_grasp.zip` + `_vecnorm.pkl`)을
실제 OMY-F3M 로봇에서 실행한다. 박스 좌표(`x, y, yaw`)를 받아 EE 를 박스 위로
정렬한 뒤 **하강 → 파지 → 들기**까지 수행한다.

> 좌표는 **지금은 수동 입력**. 나중에 카메라가 같은 `(x, y, yaw)` 자리를 채운다.

모델은 **cartesian pose 보정값** `[Δx, Δy, Δyaw]` 만 내고, 관절 변환은 MoveIt(IK)이 한다.
실제 로봇 인프라가 60Hz 폐루프를 못 돌리므로, 정책을 파이썬 안에서 가상으로 굴려
**최종 정렬 EE pose** 를 미리 계산하고 그 한 점만 MoveIt 으로 이동한다("내부 롤아웃").

---

## 0. 빠른 실행 (TL;DR)

```bash
# 환경 활성화
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/rl

# (A) 로봇 없이 단위 검증
python3 grasp_policy.py            # 정책 로드 OK?
python3 grasp_rollout.py           # status=converged?

# (B) dry-run — 로봇 안 움직임, 계획만
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0

# (C) 실제 실행 (로봇 bringup 후) — yaw 끄고 자세부터
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.0 \
    --no-align-yaw --execute --confirm EXECUTE_GRASP

# (D) 전체 (yaw 정렬 포함)
python3 run_grasp.py --box-x 0.45 --box-y -0.10 --box-yaw 0.3 \
    --execute --confirm EXECUTE_GRASP
```

기본은 **dry-run**. `--execute --confirm EXECUTE_GRASP` 둘 다 있어야 로봇이 움직인다.

---

## 1. 사전 준비

### 1-1. 환경
```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
```
필요 패키지: `stable-baselines3 2.7.1`, `torch`, `numpy`, `gymnasium` (이미 설치됨).
**IsaacLab / Isaac Sim 불필요.**

### 1-2. 로봇 bringup (별도 터미널)
```bash
ssh root@omy-SNPR44B1021.local
ros2 launch open_manipulator_bringup omy_f3m.launch.py
```

### 1-3. EE 를 박스 근처·수직파지 자세로 먼저 이동
정책이 학습한 시작 분포(박스에서 30cm 이내)에 들어가야 롤아웃이 발산하지 않는다.
`../run_ee_pose_move.py` 로 박스 근처 호버 위치로 보내 둔다.

### 1-4. 박스 좌표 측정 (수동 입력의 핵심)
1. 박스를 작업공간에 놓는다 (학습 분포: `x≈0.45±0.10, y≈-0.10±0.10`).
2. EE 를 박스 중심 **바로 위**로 이동시킨다.
3. `../show_ee_pose.py` 로 그 지점 `x, y` 를 읽어 `--box-x --box-y` 로 사용.
4. `--box-yaw` 는 처음엔 **0** (박스를 그리퍼와 평행하게 놓고) 시작.

---

## 2. 실행 — 3단계 검증 순서 (반드시 이 순서로)

자세 매핑에 미검증 리스크가 있다(sim `(0,1,0,0)` vs 실제 `[0.5063,...]`).
**작게 시작해 단계적으로 확인**한다.

### 1단계 — dry-run (로봇 안 움직임)
```bash
python3 run_grasp.py --box-x <측정값> --box-y <측정값> --box-yaw 0.0
```
**터미널에서 확인:**
- `rollout status=converged` — 미수렴이면 박스 좌표/시작 자세 재확인
- `align err: xy=[?,?] mm` — 5mm 안쪽인지
- 각 step `plan ok=True`, `JOINT5 CONSTRAINT ... within_tol=True`, `actual max_delta < 0.35`

### 2단계 — yaw 끄고 실제 실행 (자세 검증, **가장 중요**)
```bash
python3 run_grasp.py --box-x <값> --box-y <값> --box-yaw 0.0 \
    --no-align-yaw --execute --confirm EXECUTE_GRASP
```
**눈으로 확인할 움직임:**

| 단계 | 정상 | 문제 신호 → 의미 |
|------|------|------------------|
| pre_grasp | EE 가 박스 **중심 바로 위**로 와서 멈춤 | xy 치우침 → 좌표계 불일치 / grip offset |
| grasp(하강) | 그리퍼가 박스를 **양옆에서** 감싸며 내려옴 | 위에서 찍거나 옆을 침 → **top-down vs side 자세 불일치** (최대 리스크) |
| close | 짧은 변(4.4cm)을 물고 멈춤 | 헛닫힘/박스 밀림 → grasp_z 높이 / xy 오차 |
| lift | 박스가 **딸려 올라옴** | 미끄러짐/낙하 → 파지력·정렬 부족 |

자세가 명백히 틀리면 **여기서 멈추고** `config.GRIP_CENTER_OFFSET_XY` / orientation 매핑 조정.

### 3단계 — yaw 정렬 활성화
```bash
# 박스를 비스듬히(예: 20°) 놓고 그 각을 box_yaw 로
python3 run_grasp.py --box-x <값> --box-y <값> --box-yaw 0.35 \
    --execute --confirm EXECUTE_GRASP
```
**확인:** 그리퍼가 박스 회전각만큼 손목을 돌려 짧은 변에 정렬해 잡는지.
안 돌거나 반대로 돌면 → yaw 부호/회전축 매핑 문제.

---

## 3. 주요 옵션

| 옵션 | 기본 | 설명 |
|------|------|------|
| `--box-x --box-y` | 필수 | 박스 중심 (link0 기준) [m] |
| `--box-yaw` | 0.0 | 박스 yaw [rad] |
| `--no-align-yaw` | (켜짐) | yaw 정렬 끄고 수직파지 기본 자세 (검증 1단계) |
| `--no-lift` | — | 파지 후 들어올리기 생략 |
| `--approach-z / --grasp-z` | YAML | 호버/파지 높이 [m] (미지정 시 YAML waypoints `1`,`2`) |
| `--lift-offset` | 0.06 | approach_z 위로 추가 높이 |
| `--execute` | False | 실제 실행 (없으면 dry-run) |
| `--confirm` | "" | `EXECUTE_GRASP` 입력 필요 |
| `--no-step-prompts` | — | 단계별 타이핑 확인 생략 (권장 X) |
| `--force` | — | 롤아웃 미수렴/분포 경고 무시 (주의) |
| `--max-joint-delta` | 0.35 | 현재→목표 최대 관절 변위 [rad] |
| `--no-constrain-joint5` | (켜짐) | joint5=π/2 path constraint 비활성화 |

---

## 4. 안전장치

- **기본 dry-run** — `--execute --confirm EXECUTE_GRASP` 둘 다 필요
- **단계별 타이핑 확인** — 각 step 이름을 직접 입력해야 진행
- **3단계 guard** (run_pick_place 와 동일) — workspace / MoveIt plan delta / actual delta
- **롤아웃 미수렴 시 실행 거부** — 학습 분포 밖이면 멈춤
- **박스 분포 경고** — `x≈0.45±0.10, y≈-0.10±0.10, yaw ±80°` 밖이면 경고·차단
- 매 단계 **비상정지 손 닿는 곳에**. 2단계 자세 틀리면 무리해서 3단계로 가지 말 것.

---

## 5. 좌표 / 주의사항

- 박스 좌표는 **base(link0) 기준** `x[m], y[m], yaw[rad]`. `x`=앞쪽, `y`=오른쪽(음수).
- **시작 EE 를 박스 근처(30cm 이내)**, 수직파지 자세로 두고 시작. 너무 멀면 롤아웃 발산.
- z 높이(approach/grasp)는 정책과 무관 — YAML 또는 `--approach-z/--grasp-z`.
- `config.GRIP_CENTER_OFFSET_XY`: 정책은 grip center(양 finger 평균)를 정렬하는데
  실제 TF 는 link6 를 읽는다. 일관된 xy 치우침이 보이면 그 offset 을 채운다 (1차값 0).

---

## 6. 파일 구성

| 파일 | 역할 |
|------|------|
| `config.py` | 모든 상수 (sim env 에서 그대로 — **변경 금지**) |
| `grasp_policy.py` | SB3 PPO + VecNormalize 로더. `obs(6) → action(3)` |
| `grasp_rollout.py` | 내부 롤아웃. `(현재 EE, 박스) → 최종 정렬 EE pose` |
| `run_grasp.py` | 메인. 박스좌표 입력 → 롤아웃 → MoveIt 이동 → 하강/파지/들기 |
| `checkpoints/` | `motion1_grasp.zip` + `_vecnorm.pkl` (정책 — zip+pkl 세트) |

재사용: `../run_pick_place.py`(plan/실행/그리퍼 헬퍼), `../../real_moveit_common.py`(MoveIt/TF/quat).

---

## 7. 데이터 흐름 한눈에

```
박스 (x,y,yaw) ──┐
현재 EE (TF2) ───┼─▶ grasp_rollout: 정책 60Hz 가상 롤아웃 (Δ 누적) ─▶ 최종 정렬 EE (xy,yaw)
                 │                                                          │
                 └────────────────────────────────────────────────────────▼
              run_grasp → MoveIt(IK): EE pose ─▶ joint trajectory ─▶ pre_grasp→grasp→close→lift
```
