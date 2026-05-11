# motion1 — Grasp / Insert RL Task 정리

두 task 모두 `DirectRLEnv` 기반, `DifferentialIKController` (DLS) 로 cartesian Δ → joint 명령 변환.
EE 정의: 양 finger body (`rh_p12_rn_l2`, `rh_p12_rn_r2`) 월드 좌표 평균.
EE base quat: `(0, 1, 0, 0)` (수직 아래 + finger Y) + `R_z(yaw)`.

---

## 1. Action (둘 다 3차원 [-1, 1])

| 차원 | grasp | insert |
|---|---|---|
| `Δx, Δy` | actual ee xy + Δ × `0.01` (10 mm/step) | actual ee xy + Δ × `0.005` (5 mm/step, fine 정렬) |
| `Δyaw` | `ee_target_yaw + Δ × 0.05` (**누적**, ±π/2 clip) | `actual ee_yaw + Δ × 0.05` (**비누적**, ±π/2 clip) |
| ee_z | 고정 `0.17` (PRE_GRASP_Z) | 고정 `0.26` (TRANSPORT_Z) |
| gripper | open 고정 (4 joint = 0) | close 고정 (cmd 0.8 × tip_ratio 2.3) |

- xy 는 둘 다 비누적 (`actual ee + Δ`).
- yaw 만 grasp 가 누적 / insert 가 비누적 — insert 는 [insert_env.py:132-137](tasks/insert/insert_env.py#L132-L137) 에서 "xy 와 동일 패턴" 으로 변경.
- IK target = `(target_xy_w, fixed_z)` + `R_z(target_yaw) ⊗ (0,1,0,0)` → DLS IK (양 finger jacobian 평균).

---

## 2. State

### Grasp — 6차원 ([grasp_env_cfg.py:39](tasks/grasp/grasp_env_cfg.py#L39))

```
[obj_rel_x, obj_rel_y, obj_yaw_err, ee_vel_x, ee_vel_y, yaw_vel]
```

| 항목 | 계산 |
|---|---|
| `obj_rel_xy` | `box_xy_env − ee_xy_env` (env-rel) |
| `obj_yaw_err` | `wrap(box_yaw − ee_target_yaw)` |
| `ee_vel_xy` | actual finger 평균 lin vel |
| `yaw_vel` | `(ee_target_yaw − prev_target_yaw) / dt` (**target 기반**) |

### Insert — 7차원 ([insert_env_cfg.py:35](tasks/insert/insert_env_cfg.py#L35))

```
[slot_rel_x, slot_rel_y, slot_yaw_err, is_grasping, ee_vel_x, ee_vel_y, yaw_vel]
```

| 항목 | 계산 |
|---|---|
| `slot_rel_xy` | `cell_xy − ee_xy_env` (target = 셀 중심) |
| `slot_yaw_err` | `wrap(cell_yaw − actual ee_yaw)` (`_extract_ee_yaw` quat→yaw) |
| `is_grasping` | `(finger↔box dist < 7 cm) AND (box_z_env > 12 cm)` — float 0/1 |
| `ee_vel_xy` | actual finger 평균 lin vel |
| `yaw_vel` | actual finger angular vel z (`_ee_ang_vel_z`) |

### 핵심 차이

- 타겟 — grasp = 박스 / insert = 셀
- insert 만 `is_grasping` 플래그 추가 (holding 유지 학습)
- grasp 의 yaw_vel 은 **target 기반**, insert 의 yaw/yaw_vel 은 **actual ee 기반**

---

## 3. Reward (둘 다 all-positive 지향, `r_smooth` 만 음수)

### Grasp ([grasp_env.py:260-279](tasks/grasp/grasp_env.py#L260-L279))

```
r_xy_align     = exp(-80 · xy_dist²)
r_yaw_align    = exp(-5  · yaw_err²)
r_smooth       = -0.01 · (vx² + vy² + yaw_vel²)
r_success      = aligned · 50            # 매 step bonus
r_success_lump = will_succeed · 5000     # hold 30 step 도달 시 한 번
```

| 조건 | 값 |
|---|---|
| aligned | `|rel_x|<5mm & |rel_y|<5mm & |yaw_err|<0.05 rad` |
| fail | `|rel_xy| > 30 cm` |
| success | aligned 가 30 step (= 0.5초) 유지 |

### Insert ([insert_env.py:254-273](tasks/insert/insert_env.py#L254-L273))

```
r_xy_align       = exp(-80  · xy_dist²)   # 멀어도 신호 (exploration)
r_xy_align_close = exp(-200 · xy_dist²)   # 가까이 sharp (정밀 정렬, dual)
r_yaw_align      = exp(-5   · yaw_err²)
r_smooth         = -0.01 · (vx² + vy² + yaw_vel²)
r_success        = aligned · 50           # aligned = xy & yaw & is_grasping
r_success_lump   = will_succeed · 5000    # hold 15 step 도달 시
```

| 조건 | 값 |
|---|---|
| aligned | `|rel_x|<10mm & |rel_y|<10mm & |yaw_err|<0.087 (≈5°) & is_grasping` |
| fail | `box_z_env < 12 cm` (drop) **또는** `|slot_rel| > 30 cm` |
| success | aligned 가 15 step (= 0.25초) 유지 |
| drop 페널티 | **없음** — termination 으로 자연 페널티 |

### 차이 요약

- **dual exp** (gain 80 + 200) — insert 만 정밀 정렬용 추가
- **threshold 완화** — xy 5→10 mm, yaw 2.86°→5°, hold 30→15 step
- `aligned` 조건에 **`is_grasping` 마스크** 포함 (xy/yaw align 보상 자체엔 마스크 없음 — v12 에서 제거)
- **drop termination** 추가 (박스 떨어뜨리면 episode 즉시 종료)

---

## 4. 시작 분포 (reset)

| 항목 | grasp | insert |
|---|---|---|
| 박스 | spawn xy ±10cm, yaw ±80°, z=0.07 | handoff dataset 의 `box_pos_env, box_quat` 그대로 |
| 로봇 자세 | `fallback_arm_pos` + joint noise ±0.05 | handoff dataset 의 `joint_pos` (10 joint) 그대로 |
| ee_target_yaw | 0 으로 reset | dataset 의 `ee_target_yaw` (clip) |
| 셀 | — | dataset 의 `cell_xy, cell_yaw` |
| 추가 noise | env 내부 random | **없음** — dataset 자체에 transport noise (3~5 cm) 포함 |

handoff dataset = `checkpoints/insert_handoff_states.npz` (collect_insert_handoff.py 로 수집).

---

## 5. 공통 cfg

- `decimation = 1`, `sim.dt = 1/60` (insert), `sim.dt = 1/120 + decimation 2` (grasp)
- `episode_length_s = 5.0`
- `num_envs = 128`
- PPO: `n_steps=1024, batch=256, lr=3e-4, gamma=0.97, gae=0.95, clip=0.2, ent=0.005, vf=0.5, n_epochs=5`
