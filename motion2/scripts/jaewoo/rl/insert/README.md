# insert/ — insert RL 정책 sim2real (yaw-only)

motion3 에서 학습한 **insert 정책**(`checkpoints/motion3_insert_v25.zip` + `_vecnorm.pkl`)을
실제 OMY-F3M 로봇에 올리기 위한 코드. 잡은 박스를 **셀 위 호버에서 손목 yaw 만 정렬**한다.

> ⚠️ **grasp 보다 미완성**입니다. 아래 "단독 실행 한계"를 반드시 읽으세요.

---

## insert = yaw-only RL

motion3 적재 시퀀스: `grasp → lift → 뒤로 돌기 → ★insert(셀 위 yaw 정렬)★ → place 하강`.
insert 는 그중 **4단계**로, 박스를 잡은 채 셀 구멍 각도에 손목 yaw 를 맞춥니다.

- **정책 obs(7)**: `[slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel]`
- **action(3)**: `[Δx, Δy, Δyaw]` — 하지만 **yaw_only 라서 Δyaw 만 사용**. xy 는 IK 가 셀에 고정.
- **yaw 처리**: reference-anchored unwrap — 시작 그리퍼 yaw(`_yaw_ref`) ± `1.75rad` 로만 누적.
- **yaw 오차**: ±90° fold (박스 180° 대칭 → 180° 차이도 정렬로 인정).

즉 롤아웃은 **yaw 1차원만** 굴려 최종 정렬 yaw 를 계산합니다.

---

## ⚠️ 단독 실행 한계 (grasp 와 결정적으로 다른 점)

grasp 는 "박스 위로 가서 잡기"라 단독 실행이 됩니다. **insert 는 안 됩니다** — 다음 3가지 때문:

1. **앞단 선행 필요** — insert 는 "박스를 **잡고**, 뒤로 **돌려**, 셀 위 **호버**"가 된 상태에서
   시작합니다. 그 앞단(grasp→lift→turn)은 이 폴더 밖이라 별도로 만들어야 합니다.
   `run_insert.py` 는 그 호버 자세가 이미 됐다고 **전제**합니다.

2. **뒤쪽 좌표계 / workspace** — 셀은 로봇 **뒤쪽(-x)** 에 있습니다. motion2/jaewoo 의 기본
   workspace(`x_min=-0.10`)와 수직파지 인프라는 **앞쪽(+x)** 기준이라 그대로 안 맞습니다.
   `run_insert.py` 에서 `--x-min` 등으로 workspace 를 셀 위치에 맞게 조정해야 합니다.

3. **place 하강 드리프트** — insert RL 이 yaw 를 잘 맞춰도, 그 뒤 셀로 **하강할 때 xy 가
   8~10cm 드리프트**하는 게 sim 에서도 **미해결 병목**입니다(IK/모션 문제, RL 아님).
   실물에선 더 클 수 있으니 `--place-z` 하강은 충분히 검증 후에만.

**정리**: 이 폴더는 insert 정책 추론 + yaw 롤아웃까지는 **검증 완료**. 하지만 실제 로봇에서
끝까지 돌리려면 ① 앞단 chain(grasp→turn) 통합, ② 뒤쪽 좌표계 정의, ③ 하강 드리프트 대응이
추가로 필요합니다.

---

## 빠른 실행

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate env_isaaclab
cd /home/jaewoo/IsaacLab/source/motion2/scripts/jaewoo/rl/insert

# (A) 로봇 없이 단위 검증 (이미 통과 확인됨)
python3 insert_policy.py        # 정책 로드 OK?
python3 insert_rollout.py       # status=converged? (yaw 정렬)

# (B) dry-run — 박스 잡고 셀 위 호버 자세에서 (로봇 안 움직임)
python3 run_insert.py --cell-x <셀x> --cell-y <셀y> --cell-yaw 0.0

# (C) 실제 실행 — align(yaw 회전)만, place 하강 생략
python3 run_insert.py --cell-x <셀x> --cell-y <셀y> --cell-yaw 0.0 \
    --x-min -0.55 --execute --confirm EXECUTE_INSERT
#   (셀이 뒤쪽이면 --x-min 음수로 workspace 확장)

# (D) place 하강까지 (⚠️ 드리프트 검증 후)
python3 run_insert.py --cell-x <셀x> --cell-y <셀y> --cell-yaw 0.0 \
    --place-z 0.07 --x-min -0.55 --execute --confirm EXECUTE_INSERT
```

기본 dry-run. `--execute --confirm EXECUTE_INSERT` 둘 다 필요.

---

## 파일 구성

| 파일 | 역할 |
|------|------|
| `config.py` | insert 규약 상수 (yaw_only, yaw_margin 1.75, ee_z 0.20 등 — **변경 금지**) |
| `insert_policy.py` | SB3 PPO + VecNormalize 로더. `obs(7) → action(3)` |
| `insert_rollout.py` | yaw-only 롤아웃. `(셀 yaw, 시작 ee yaw) → 최종 정렬 yaw` |
| `run_insert.py` | ⚠️ 실험적. 호버 yaw 정렬 + (옵션)place 하강 |
| `checkpoints/` | `motion3_insert_v25.zip` + `_vecnorm.pkl` |

재사용: `../../run_pick_place.py`(plan/실행 헬퍼), `../../../real_moveit_common.py`(MoveIt/TF/quat).

---

## grasp 와 차이 한눈에

| | grasp | insert |
|---|---|---|
| obs | 6 | 7 (is_grasping 추가) |
| 제어 | xy + yaw | **yaw only** (xy 는 IK 고정) |
| yaw | 단순 누적 | reference-anchored unwrap (±1.75rad) |
| 단독 실행 | ✅ 가능 | ❌ 앞단(grasp→turn) 선행 필요 |
| 좌표계 | 앞쪽(+x) | 뒤쪽(-x) — workspace 조정 필요 |
| 추가 난관 | — | place 하강 xy 드리프트 (미해결) |
