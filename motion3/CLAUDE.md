# CLAUDE.md — motion3 (책상 위 로봇 + 뒤쪽 저고도 적재)

> 다음 세션이 즉시 컨텍스트 잡고 이어가도록 작성. motion1을 복사해 만든 **적재 재설계** 프로젝트.
> 전체 진행 플랜: `/home/jaewoo/.claude/plans/frolicking-mixing-mitten.md`

---

## ⚡ TL;DR — 현재 상태 (2026-06-10 v25 갱신)

**해결됨. 근본원인 = 절대 yaw setpoint의 wrap 경계(±π) 함정. 해법 = reference-anchored unwrap(v25): setpoint를 reset 시 측정 grip yaw(`_yaw_ref`)에 anchor하고 `_yaw_ref ± yaw_margin(1.75rad)`로만 누적 제한 → 고정 wrap 경계 소멸. 결과 success 0.05(v24)→0.57(v25), yaw_aligned 0.035→0.58. ① chain parity 수정 완료(stage_rl_insert: xy=cell 고정, yaw=누적 setpoint anchor) — env와 동일 convention.**

### v25 결과 (2026-06-10, `motion3_insert_v25.zip`)
| metric | v23 | v24(±π clip) | **v25(anchor)** |
|---|---|---|---|
| success/aligned | 0.26 | 0.05 | **0.57** |
| yaw_aligned(5°) | 0.18 | 0.035 | **0.58** |
| yaw_err_abs_mean | 30° | 8.5° | **10.9°(0.19rad)** |
| xy_dist_mean | 4mm | 4mm | 8.9mm |
- xy 4→8.9mm 증가: 정책이 yaw를 적극 회전→그 coupling이 xy 끌어냄(예상된 trade). 여전히 10mm 임계 안(xy_aligned 0.76).
- 다음: play_insert 육안확인(joint6 flip 사라졌는지) + chain end-to-end(parity) 검증.

### 핵심 설계 전환 (v23까지 적용 완료)
- **insert = yaw-only RL**: xy는 IK가 cell에 고정 holding(`cfg.yaw_only=True`, `_pre_physics_step`에서 xy target 미갱신), RL은 **yaw만** 학습. → xy_aligned 0.96/xy_dist **4mm** (xy 완전 해결). 이유: yaw를 적극 IK 제어하면 그 회전이 grip center xy를 끌어내는 coupling 때문에 yaw+xy 동시 불가 (v16~v20에서 xy 0.10 plateau로 확정). xy는 motion/IK가 잡고 RL은 yaw만.
- **yaw target 누적 복구**: `_ee_target_yaw += delta`(영속 setpoint). 이전 비누적은 회귀버그였음.
- **기하 수정 (관통 버그)**: 옛날 hover 0.20은 박스 밑면 0.04로 셀 벽(0.12) 7.7cm 관통. → collect의 **하강(3d-2) 제거**, `INSERT_HOVER_Z=LIFT_Z(0.56)`로 올림 → handoff를 turn+lift 높은 위치에서 저장(박스 밑면 0.36, 벽 위 24cm). 실제 셀 하강/place는 chain runner 전담.
- **방식 B**: collect의 3d-1 이동을 **xy만**(yaw는 turn된 자연값 유지), handoff에 **실제 그리퍼 yaw** 저장 → RL이 자연 yaw 오차 보정. reset_yaw_noise=0(자연오차로 충분).
- **box spawn yaw ±80°→±50°** (90° 근처 fold경계 불가케이스 감소). 박스 **대칭(방향성 없음, 사용자 확정)** → fold 유지 OK.
- 데이터: **`insert_handoff_states_v16.npz` (2734개, 3000 수집→슬립 266 필터)**. cfg `handoff_dataset_path`=v16.

### ★ 진짜 yaw 근본원인 (2026-06-10 확정 — 이게 결론)
- v16 handoff의 `ee_target_yaw`(그리퍼 yaw)가 **100% ±90°~±180°**(turn=joint1 회전으로 그리퍼가 뒤를 향함). 그런데 `ee_yaw_min/max` clip이 **±90°**였음.
- → reset 순간 setpoint=clamp(±180°,±90°)=±90° ≠ 그리퍼(±180°) → IK가 그리퍼를 90~180° **홱 회전**(=사용자가 본 "joint6 휙 돌림") + 매 에피소드 그 회전에 시간낭비 → **yaw success 0.26(v23) 한계**.
- v21(낮은 0.20, yaw 0.85)이 괜찮았던 건 그땐 method A(cyaw 저장, ±90° 안)였기 때문. method B(실제 ±180° 저장)+±90°clip이 겹쳐 망가짐. **즉 "높이 0.56 때문"이 아니라 clip 불일치가 주범.**
- **수정 → v24 결과(2026-06-10 완료): `ee_yaw ±π` 적용.** `yaw_err_abs_mean 30°→8.5°`(flip 사라지고 일관되게 근접 — **clip이 근본원인 확정**). **BUT `yaw_aligned(5°내) 0.18→0.035, success 0.26→0.05`로 오히려 낮음** — 8.5°에 모여있지만 5° 밑으로 못 조임(±180° wrap 경계 진동 추정). xy는 여전히 완벽(4mm).
- **v24 play_insert 육안확인(사용자, 2026-06-10): 정책이 yaw를 거의 안 돌림(수동적).** 즉 ±π clip이 flip은 없앴지만 **±180° wrap 경계에서 yaw action이 불안정 → 학습이 "yaw 돌려도 보상 안정적 증가 X"로 판단해 yaw 조정 포기**(success 0.05 < v23 0.26). yaw_err 8.5°는 적극정렬이 아니라 거의-정렬 시작값에 머문 것.
- **★해결(v25) = reference-anchored unwrap (unwrap 누적 채택).** `insert_env`: `_yaw_ref` 버퍼 추가, reset 시 `_yaw_ref=_ee_target_yaw=ee_target_yaw_d(측정 grip yaw)`, `_pre_physics_step`에서 `clamp(_yaw_ref-m, _yaw_ref+m)` (m=`cfg.yaw_margin`=1.75). 절대 ±π clamp 폐기 — quat은 |yaw|>π 연속이라 margin이 yaw_ref와 함께 움직여 고정 wrap 경계가 사라짐. margin은 fold 한계 90°를 덮어야(자연오차 최대 83°) 정상 보정 안 막음. → success 0.57(위 표). **±π clip 단독(v24)은 답 아니었음**(flip은 고쳤으나 학습 망침) — anchor가 핵심.

### 폐기된 가설들 (시간 낭비 방지 — 다시 파지 말 것)
- "reward 문제" ✗ (cone/gain 튜닝 다 효과 없었음). "측정 프레임 90° offset" ✗ (gripper==box yaw). "손목 joint6 직접제어 필요" ✗ (probe가 grip 느슨해 box drop으로 오판, 실제론 누적복구로 IK가 yaw 추종함). "hover 0.30으로 낮춰야" △ (clip이 진짜 원인이라 height 영향 작을 수도 — v24로 확인).

### ckpt 이력
**`motion3_insert_v25.zip`=현 최선(reference-anchored unwrap, success 0.57, yaw_aligned 0.58, xy 8.9mm).** v24=±π clip(0.05, 학습망침). v23=±90°clip(0.26). v21=옛 낮은기하. v16~v20=실패 이력. chain 기본 insert_checkpoint는 아직 `motion3_insert.zip` — **v25로 갱신 필요**(또는 chain --insert_checkpoint로 지정).

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

### ★ place 하강 드리프트 — 현재 미해결 병목 (2026-06-10)
- **증상**: insert RL은 grip xy를 cell에 **0.4cm로 완벽 정렬**(yaw도 ee 0.4~2.4°). 그런데 그 뒤 **place 깊이로 하강하면 grip xy가 -x(뒤)로 ~8~10cm 드리프트**(near·far 셀 모두). 적재 실패의 진짜 원인 = yaw·RL 아니라 **하강 motion/IK**.
- **진단(chain xy 로그 [xy@RL-end]/[xy@descend] 8+5 run)**: 드리프트는 **하강 거리에 비례하는 lag 아니라 endpoint 문제** — RL hover를 0.56→0.32로 낮춰 하강 49→25cm로 줄여도 드리프트 ~10cm 그대로. 즉 도착점(z≈0.07 place 깊이)에서 IK가 그 xy를 못 잡음(팔이 뻗으며 EE가 -x로 미끄러짐). DLS damping under-reach + orientation-lock(수직down 고정) 합성. **reach 거리 가설은 반박**(near 셀도 동일 드리프트).
- **A 적용(저-hover, v25 재사용·재학습X, 사용자 확정 "A로")**: chain `RL_HOVER_Z`(layout 무변경, TRANSPORT_Z만 override) + 5a insert duration 1.5→2.5. 드리프트 자체는 안 줄지만 **박스를 셀에 더 가까이 놓아 벽 funneling** → 최종 적재 개선.
- **hover-sweep 결과 (RL_HOVER_Z 튜닝)**: **0.32 = sweet spot** (4 run 최종 1~5cm, 무사고). **0.28 로 더 내리면 박스가 벽 모서리에 걸려 텀블 — run4 33cm/box yaw 58° 카타스트로피**(near 일부는 2cm로 개선되나 위험). **0.56(원본) = 0.2~18cm 들쭉날쭉**. ★벽(0.12)이 하드 제약이라 hover를 0.32 밑으로 못 내림 + grip descend 드리프트(~10cm)는 어느 hover에서도 동일(endpoint). near 셀 ~5cm 가 hover-only 접근의 바닥.
- **near 셀 sub-5cm 하려면 hover 아닌 다른 레버 필요**(closed-loop 하강+벽인지 xy / place 전용 arm config / box-drop+tip제어) — probe(place reach) 검증 필요한데 env wedge 로 보류.
- **남은 레버(미적용, A와 직교)**: near 셀 더 조이려면 place-깊이 reach 개선 — (a) 셀 당김, (b) TABLE_HEIGHT↑(far hover reach 트레이드), (c) box-drop(낙하 품질 risk), (d) place 전용 arm seed/config, (e) closed-loop 하강. 에이전트 2(ik-motion-planner/Plan) 논의: 모션패치(seed/느린하강) vs 셀당김 — 데이터는 endpoint라 셀당김 효과 제한적.
- yaw 로그: ee yaw 하강 중에도 0.4~2.4° 유지(안 틀어짐). box yaw는 0.9~16.6°(벽 접촉·안착 시 회전) — RL/yaw는 병목 아님 확정.

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

## 4. 다음 할 일 (우선순위) — 2026-06-10 (v25 갱신)

1. ✅ **done — v25(reference-anchored unwrap) 학습**: success 0.57(위 표). 코드: insert_env(`_yaw_ref` anchor + `yaw_margin` clamp), insert_env_cfg(`yaw_margin=1.75`).
2. ✅ **done — chain runner parity 수정** (`stage_rl_insert`): 버그A(비누적+±90°clip→누적 anchor `ee_target_yaw_acc`, stage 진입 시 `quat_z_yaw(grip)`로 init, `clamp(yaw_ref±RL_INSERT_YAW_MARGIN=1.75)`), 버그B(정책 xy 버리고 xy=cell 고정). env yaw_only convention과 일치. **검증중**: `/tmp/chain_v25.log` (repeat 5, v25 ckpt) — insert success / 최종 box-cell 거리 확인.
2.5. ✅ **done — chain yaw 주체를 RL 로 이관 (2026-06-10, 사용자 지적)**: 이전엔 3d-1 이 **정확한 cyaw 로 yaw 를 pre-align** 해 RL 전에 yaw 가 다 맞아 RL 우회(=chain 에서 yaw 즉시성공은 motion 덕, RL 아님). sim 에선 되나 **실물은 cyaw 가 비전추정이라 무너짐**. → 3d-1/3d-2 를 **자연 yaw(turned_quat) 유지**로 바꾸고, **stage 4 RL 이 yaw 보정**, RL 최종 yaw(`insert_final_yaw`)를 downstream(5a/5b/6a) `cell_ee_quat` 으로 사용(motion 이 다시 cyaw 로 snap 안 함). branch 선택(cyaw/cyaw+π)은 RL obs 의 fold_yaw_sym 이 자동 처리 → 제거. **결과 chain yaw 성공은 이제 v25 실제치(~0.57) 반영**(이전 인플레된 ~1.0 아님 — 정직/sim2real 충실). 미검증: 재실행 육안확인 필요.
3. **★ 다음**: (a) play_insert 육안확인 — `./isaaclab.sh -p source/motion3/scripts/play_insert.py --checkpoint checkpoints/motion3_insert_v25.zip --vecnorm checkpoints/motion3_insert_v25_vecnorm.pkl --episodes 30` 로 joint6 flip 사라지고 yaw 적극 정렬 확인. (b) chain 기본 ckpt를 v25로 갱신(`cp` 또는 default 인자 변경). (c) success 0.57 더 올릴지(yaw threshold/reward gain 추가 튜닝 or xy coupling 완화).
4. **데모 정리**: chain `--demo_grid`(이미 구현 — 고정그리드+셀 0→9 순차+박스 누적, `PlacedBox0~9` kinematic). parity 검증 후 `--demo_grid --repeat 10 --hold_s -1`로 깨끗한 적재 데모.
5. (선택) 카메라·비전(motion2 `inference/` 재사용), real robot(motion2/jaewoo/rl) 배포.

**주의:** turn-around=joint1만(§1 변경금지)은 유지. insert는 yaw-only(xy=IK 고정)로 확정 — 손목 직접제어는 폐기(누적복구로 IK가 yaw 추종됨).

**신규/수정 파일(2026-06-10):** `scripts/preview_grid_filled.py`(5×2 박스 정적 씬), `scripts/probe_yaw_{frame,joint6,verify}.py`, `scripts/probe_waypoint_fk.py`. insert_env(yaw_only/누적/reset노이즈/±π clip), insert_env_cfg(yaw_only=True, reset_yaw_noise=0, ee_yaw ±π, handoff=v16, reward gxy40/yaw_lin0.4/gain50), layout(INSERT_HOVER_Z=LIFT_Z, BOX_SPAWN_YAW_MAX=0.873), collect(3d-2제거+방식B+flush), chain(--demo_grid).

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
