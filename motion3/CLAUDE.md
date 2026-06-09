# CLAUDE.md — motion3 (책상 위 로봇 + 뒤쪽 저고도 적재)

> 다음 세션이 즉시 컨텍스트 잡고 이어가도록 작성. motion1을 복사해 만든 **적재 재설계** 프로젝트.
> 전체 진행 플랜: `/home/jaewoo/.claude/plans/frolicking-mixing-mitten.md`

---

## ⚡ TL;DR — 현재 상태 (2026-06-09 갱신)

**insert success 0%의 진짜 원인 = yaw가 reward 문제도 측정 문제도 아니라 "제어 권한 부재" + "회귀 버그(비누적 target)". 다음 = 손목 직접 yaw 제어 구현 + yaw 노이즈 주입 + 재학습.**

### 진단 결론 (probe로 확정 — `scripts/probe_yaw_frame.py`, `probe_yaw_joint6.py`)
- handoff dataset(v15)는 **정상**: box가 cell에 물리적으로 정렬됨 `|fold(cell−box_yaw)|=mean 2.1°, 95%<5°` (ground-truth=박스 quat 직접). collect의 transport(3d-1)가 모션플래닝으로 yaw 정렬 후 저장하기 때문.
- `gripper_yaw(_extract_ee_yaw) == box_yaw` (offset 0.26°). 측정 정확, rigid grasp.
- 그런데 env.step에서 yaw가 **reset 2° → 에피소드 끝 ~80°로 드리프트**(zero-action 3 step만에 18°). yaw action +max ×20 step → Δyaw chaotic(std 66°, ±75°). → IK가 insert hover 자세에서 yaw 제어/유지 불가.
- v16 학습(선형 cone reward + Gaussian gain50)도 **xy_aligned 0.02→0.38 잘 배움, yaw_aligned 0.006 그대로** → reward로 안 고쳐짐 확정.

### 근본 원인 2가지
1. **회귀 버그 — 비누적 yaw target**: grasp_env는 `_ee_target_yaw += delta`(영속 setpoint, 복원력 O, **작동**). insert_env는 `_ee_target_yaw = 측정현재yaw + delta`(매 step 재계산, 복원력 X → 드리프트 래칫). "xy와 동일 패턴"이라며 바꾼 게 grasp의 작동 패턴을 깸. [insert_env.py:138]
2. **자세 ill-conditioning**: grasp는 앞/높음(IK yaw OK), insert는 뒤/낮음 풀-리치(rotation-about-vertical near-singular → IK chaotic).

### 합의된 수정 방향 (사용자 확정 — sim2real 강건성 목표)
- yaw도 **진짜 RL 보정 대상**으로 간다 (freeze 안 함). 따라서 **둘 다 필요**:
  1. **손목 직접 yaw 제어**: 위치(xy,z)=기존 IK 유지, yaw=policy Δyaw를 **joint6(손목 roll)에 직접 누적(영속 setpoint)**. 수직-아래 자세에서 joint6=수직축 yaw. → IK ill-conditioning 우회 + 복원력 확보.
  2. **yaw 노이즈 주입**: env reset에서 손목을 랜덤 Δ로 돌려 박스째 yaw 오차 생성(재수집 불필요, 매 에피소드 다양). collect보다 reset 주입이 유리.
- 검증: 외부 probe는 grasp 우회로 박스가 떨어져 신뢰불가 → **env.step 내부(정상 grasp 유지 경로)에서 구현·검증**.
- 적용해둔 v1 reward(`reward_yaw_lin_w=2.0`, `reward_yaw_align_gain=50`)는 무해하나 효과 없음 — 손목제어 넣은 뒤 재튜닝.

이전 실패: 첫 학습은 `is_grasping=0` → 박스 위에서 잡아 매달려 box_z 임계 미달 → 완화(box_drop_z 0.085, grasping_dist 0.12)로 해결. ckpt `motion3_insert_v15a_isgrasp0_failed.zip`. yaw 실패 학습 = `motion3_insert.zip`(v15), `motion3_insert_v16.zip`(cone reward, yaw 여전히 실패).

---

## 1. 설계 (motion1 hybrid 계승)

큰 운동(reach/transport/place) = **motion planning(IK)**, contact-rich(grasp/insert) = **RL**. 단 **grasp는 motion1 정책 재사용**(`checkpoints/motion1_grasp.zip`, 재학습 안 함), **insert만 새로 학습**.

**시퀀스** (앞에서 잡아 뒤로 돌려 적재):
```
1. 다가감(PRE_GRASP_Z) → 2. grasp RL(xy/yaw 정렬) → 3a 하강(GRASP_Z) →
3b 잡기 → 3c 들기(LIFT_Z) → 3c2 뒤돌기(joint1만 회전) →
3d-1 높은데서 yaw 정렬 → 3d-2 수직 하강(호버 INSERT_HOVER_Z) →
4. ★insert RL(호버에서 xy/yaw 정밀 정렬)★ → 5 place 하강(PLACE_Z) → open
```

**좌표계** (`layout.py` 단일출처, env-relative, ground=z0):
- ground(z=0) = **뒤쪽 적재 셀 바닥**, robot base/책상 상판 = z=**TABLE_HEIGHT(0.30)**.
- 앞쪽(grasp/lift) = 책상 평면(+H), 뒤쪽(insert/place) = ground 프레임.
- 박스 **서있는 형태** `(0.118 앞뒤, 0.044 좌우-잡는변, 0.139 높이)`. 벽(0.12)보다 높이 솟는 게 정상.
- 셀 **5좌우 × 2깊이 = 10칸**, 뒤쪽(-x) 중심 (-0.38, 0). 셀 깊이(0.16)가 앞뒤. "거의 일자" → cell yaw ±10°.
- **박스 180° 대칭** → insert yaw 오차 ±90° fold (`fold_yaw_sym`), transport도 cyaw/cyaw+π 가까운 쪽 선택.

**검증된 핵심 결정 (변경 금지)**:
- IK: `DifferentialIKController(dls)`, EE=양 finger 평균, base_ee_quat=(0,1,0,0)(수직아래).
- **turn-around = joint1만 회전**(joint2~6은 lift 자세 유지) → 그리퍼 수직 유지 → joint5/6 안 돎. (down-align 불필요.)
- grasp 중 **joint1 클램프** `GRASP_JOINT1_CLAMP=(-1.0, 0.9)` — 베이스 큰 swing 방지(yaw는 joint6).
- **S1 IK 게이트 54/54 통과** (turn-around seed로 뒤쪽 10칸 도달 확인, H=0.30/center=-0.38). reach band ≈ 반경 0.27~0.50.
- z값: PRE_GRASP 0.55, GRASP **0.455(사용자 확정 — 더 깊이 잡으면 안 됨)**, LIFT 0.56, INSERT_HOVER 0.20, PLACE 0.07.

---

## 2. 파일 (정리 후)

**핵심 (단일출처/env)**:
| 파일 | 역할 |
|---|---|
| `layout.py` | **모든 좌표/z/grid/박스/클램프 상수 단일출처** (isaac 무의존). 여기만 바꾸면 cfg/collect/chain 자동 반영 |
| `robot_cfg.py` | `OMY_TABLE_MOUNTED_CFG` (base를 z=TABLE_HEIGHT로) |
| `scene_helpers.py` | 3×3/5×2 grid 격벽 기하 + 셀 좌표 (pure, layout 기반) |
| `tasks/insert/insert_env.py` `_cfg.py` | insert RL env. obs7/action3, ee_z 고정, yaw ±90° fold. **action/obs/reward 식은 motion1 계승** |
| `tasks/grasp/` | grasp env (정책 재사용 — env 코드 자체는 chain이 import 안 하지만 참고용 보존) |

**스크립트**:
| 파일 | 역할 |
|---|---|
| `scripts/collect_insert_handoff.py` | grasp+motion으로 호버 handoff 상태 수집 → v15.npz |
| `scripts/train_insert.py` | insert RL PPO 학습 |
| `scripts/play_insert.py` | 학습 정책 단독 시각화 |
| `scripts/play_motion_chain_with_grasp_insert.py` | **전체 통합 chain (S10)** — collect와 동일 모션 + insert RL inference |
| `scripts/probe_back_reach.py` | S1 IK 도달성 검증 (뒤쪽 셀) |
| `scripts/probe_table_height.py` | S0 책상 USD 높이 측정 |
| `scripts/preview_scene.py` | 초기 scene 미리보기 (정책 불필요) |
| `scripts/teleop_joints.py` | 관절 수동 조종 (자세 잡기) |
| `scripts/sim_watchdog.sh` | 백그라운드 sim 행/초과 자동 kill |

**삭제 권고(미실행 — 권한 거부됨, 아래 §6 명령 참고)**: `scripts/{collect_insert_coded, play_motion_chain, play_motion_chain_with_grasp, play_motion_chain_with_grasp_insert_camera, collect_yolo_dataset}.py`, `tasks/insert_v2/`, `yolo_dataset/`, `PLAN.md`, `RL_TASKS_SUMMARY.md`, `waypoiny.yaml`. (비전 파일은 motion1 scene 하드코딩이라 motion3 무용 — 미래 카메라는 motion2 `inference/`(unproject/yolo) + layout.py 기반 신규 작성. motion1 원본에 카메라 link6 부착 패턴 보존됨.)

---

## 3. 실행 (conda 활성화 필수 — 백그라운드도)

```bash
source /home/jaewoo/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
# scene 미리보기
./isaaclab.sh -p source/motion3/scripts/preview_scene.py
# dataset 수집 (headless, ~5초/개)
./isaaclab.sh -p source/motion3/scripts/collect_insert_handoff.py --headless --target 3000
# insert RL 학습 (~3분, SPS 10000+)
./isaaclab.sh -p source/motion3/scripts/train_insert.py --headless --num_envs 128 --timesteps 1500000 --name motion3_insert
# 정책 시각화
./isaaclab.sh -p source/motion3/scripts/play_insert.py
# 전체 통합 (학습 후)
./isaaclab.sh -p source/motion3/scripts/play_motion_chain_with_grasp_insert.py --hold_s 5 --repeat 5
```
백그라운드 실행 시: `export OMNI_KIT_ACCEPT_EULA=YES; ... < /dev/null` + `sim_watchdog.sh <PID> <MAX> <log> <STALL>` 동반. (collect STALL은 ≥900s — partial save 사이 조용함.)

---

## 4. 다음 할 일 (우선순위) — yaw 제어 재설계

1. **★ 손목 직접 yaw 제어 구현** (`insert_env.py`):
   - `_pre_physics_step`: yaw target을 **누적**으로 복구 (`_ee_target_yaw += delta_yaw`, grasp 패턴). 비누적 제거.
   - `_apply_action`: 위치 IK는 유지하되 yaw를 **joint6 직접 오프셋**으로. (a) IK는 고정 reference yaw(예: 0 또는 cell_yaw)로 수직자세 풀고, (b) 최종 `joint6_target = ik_joint6 + (_ee_target_yaw − reference)` 식으로 손목에 yaw 누적. 또는 grasp처럼 GRASP_JOINT1_CLAMP 적용해 IK가 손목으로 yaw 풀게.
   - **검증**: env.step 안에서 Δyaw 명령에 box_yaw가 깨끗이(monotonic, 박스 유지) 추종하는지. joint6 아니면 인덱스만 교체.
2. **yaw 노이즈 주입** (`_reset_idx`): handoff joint_pos 로드 후 손목 관절을 `±YAW_NOISE`(예: ±30~80°) 랜덤 회전 → 박스째 yaw 오차. obs/reward는 box(=gripper) yaw vs cell_yaw로 측정(이미 그러함).
3. **reward 재튜닝**: 손목제어로 yaw 제어 가능해지면 v1 cone reward로 충분한지 확인, 부족하면 조정. → 재학습(~3분) → play_insert success 확인.
4. **S10 chain 통합 검증**: `play_motion_chain_with_grasp_insert.py` — grasp→turn→insert→place 전체. **단 chain runner도 insert 단계 yaw 제어를 손목 직접제어로 동일하게 맞춰야 함**(env와 control parity).
5. (선택) GRASP_Z 기하 / 미래 카메라·비전(motion2 `inference/` 재사용).

**주의:** 손목 직접제어는 "turn-around=joint1만, joint5/6 안 돎"(§1 변경금지)과 의도적으로 다름 — insert 단계 한정 yaw 제어용. turn-around 자체는 그대로.

---

## 5. 주의 / 함정

- **frozen (수정/덮어쓰기 금지)**: `source/motion1/` 전체, `source/example5/`, `source/omy/`, `OMY.usd`, `checkpoints/{example5,motion1_grasp,*_best,*_v14*,*_failed}.zip`, `robotis_lab/`(읽기만).
- 백그라운드 sim은 **conda 미활성 시 startup crash/hang**. 반드시 conda + EULA + stdin /dev/null. 행나면 watchdog가 kill.
- 새 잡 전 `nvidia-smi` + `pgrep -af kit` 로 GPU/고아 프로세스 확인 (고아 kit가 GPU 잡으면 다음 잡 행).
- grasp 정책 obs는 상대량(z 무관) → robot elevated 해도 재사용 OK. 단 절대 z 타깃은 layout에서 +H.
- 커스텀 에이전트(rl-reward-tuner 등 `.claude/agents/`)는 세션 재시작해야 스폰됨. 안 되면 general-purpose에 전문지식 주입.

---

## 6. 정리(삭제) 명령 — 사용자가 실행 (내 rm 권한 거부됨)

```bash
cd /home/jaewoo/IsaacLab/source/motion3
rm -f scripts/collect_insert_coded.py scripts/play_motion_chain.py \
      scripts/play_motion_chain_with_grasp.py \
      scripts/play_motion_chain_with_grasp_insert_camera.py \
      scripts/collect_yolo_dataset.py
rm -rf tasks/insert_v2 yolo_dataset
rm -f PLAN.md RL_TASKS_SUMMARY.md waypoiny.yaml
find . -name __pycache__ -type d -exec rm -rf {} +
```
(grasp 관련 `tasks/grasp/`, `train_grasp.py`, `play_grasp.py`는 **보존** — 사용자 요청.)

---

## 7. 핵심 수치 (cheatsheet)

```
TABLE_HEIGHT=0.30  BOX_SIZE=(0.118,0.044,0.139) 서있음
PRE_GRASP_Z=0.55  GRASP_Z=0.455  LIFT_Z=0.56  INSERT_HOVER_Z=0.20  PLACE_Z=0.07
grid: 5좌우×2깊이=10칸, center=(-0.38,0), 셀 inner (앞뒤0.16 × 좌우0.065), wall 0.12
cell_yaw ±10°, cell xy noise ±3cm, transport-end ee noise 3~5cm
insert cfg: action_scale_xy 0.005, yaw 0.05, ee_fixed_z 0.20
  align thr: xy 10mm, yaw 5°(0.087), hold 15 step
  reward: xy_gain 80, xy_close 200, yaw_gain 5(↑필요), success_bonus 50, lump 5000
  is_grasping: dist<0.12, box_z>0.085  (서있는 박스 매달림 반영)
  yaw: 비누적 + ±90° fold (박스 180° 대칭)
GRASP_JOINT1_CLAMP=(-1.0, 0.9)
conda: env_isaaclab | GPU RTX4070 | grasp 정책 재사용 motion1_grasp.zip
```
