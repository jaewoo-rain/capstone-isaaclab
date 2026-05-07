from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import DirectRLEnv

from .lift_env_cfg import LiftEnvCfg
from isaaclab.utils.math import quat_apply

class LiftEnv(DirectRLEnv):
    cfg: LiftEnvCfg

    def __init__(self, cfg: LiftEnvCfg, render_mode: str | None = None, **kwargs):
        # 부모 클래스(DirectRLEnv) 초기화 — 씬 생성, 물리 엔진 시작 등이 여기서 일어남
        # _setup_scene()도 내부적으로 super().__init__ 안에서 호출됨
        super().__init__(cfg, render_mode, **kwargs)

        # -----------------------------
        # Action / Observation space
        # SB3(PPO)가 환경의 입출력 형태를 알 수 있도록 공간을 정의
        # -----------------------------

        # action: 네트워크가 출력하는 벡터
        # low=-1.0, high=1.0 → 각 관절 명령이 -1~1 사이 값
        # shape=(7,) → arm 6개 + gripper 1개
        self.action_space = gym.spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.cfg.action_space,),
            dtype=np.float32,
        )

        # observation: 네트워크가 입력받는 벡터
        # -inf~inf → 정규화는 VecNormalize가 외부에서 담당
        # shape=(34,) → cfg에서 정의한 34차원
        self.observation_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.cfg.observation_space,),
            dtype=np.float32,
        )

        # -----------------------------
        # joint / body 이름 매핑
        # -----------------------------

        # arm joint 이름 리스트: ["joint1", "joint2", ..., "joint6"]
        # f"joint{i}" → i=1~6을 문자열로 만들어서 리스트 생성
        self.arm_joint_names = [f"joint{i}" for i in range(1, 7)]

        # cfg에 정의된 gripper joint 이름 4개를 리스트로 복사
        self.gripper_joint_names = list(self.cfg.gripper_joint_names)

        # -----------------------------
        # joint limit
        # soft_joint_pos_limits: shape = [num_envs, num_joints, 2]
        # [0]       → 첫 번째 환경 (모든 환경이 동일하므로 0번만 읽음)
        # [:, 0]    → 모든 관절의 하한값
        # [:, 1]    → 모든 관절의 상한값
        # -----------------------------
        self.robot_dof_lower_limits = self._robot.data.soft_joint_pos_limits[0, :, 0].to(self.device).clone()
        self.robot_dof_upper_limits = self._robot.data.soft_joint_pos_limits[0, :, 1].to(self.device).clone()

        # -----------------------------
        # arm joint ids
        # USD에서 joint 이름 → 내부 인덱스(숫자)로 변환해서 저장
        # 나중에 텐서에서 해당 관절만 뽑을 때 이 인덱스를 사용
        # -----------------------------
        self.arm_joint_ids: list[int] = []
        for name in self.arm_joint_names:
            # find_joints 반환값: (list[int], list[str])
            # [0] → 인덱스 리스트만 꺼냄
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY arm joint를 찾지 못함: {name}")
            # found[0] → 같은 이름이 여러 개일 수 있어서 첫 번째만 사용
            self.arm_joint_ids.append(int(found[0]))

        # -----------------------------
        # gripper joint ids
        # mimic 구조상 4개를 같은 명령으로 제어
        # 대표 관절은 첫 번째를 사용
        # -----------------------------
        self.gripper_joint_ids: list[int] = []
        for name in self.gripper_joint_names:
            found = self._robot.find_joints(name)[0]
            if len(found) == 0:
                raise RuntimeError(f"OMY gripper joint를 찾지 못함: {name}")
            self.gripper_joint_ids.append(int(found[0]))

        # gripper joint 4개 중 첫 번째 → 닫힘 상태 측정의 대표값으로 사용
        # rh_r1_joint 가 master, 나머지 r2/l1/l2는 mimic
        self.main_gripper_joint_id = self.gripper_joint_ids[0]
        self.mimic_gripper_joint_ids = self.gripper_joint_ids[1:]  # r2, l1, l2

        # observation에서 사용할 joint id 목록
        # arm 6개 + gripper 4개 = 10개 인덱스를 하나의 리스트로 합침
        self.all_obs_joint_ids = self.arm_joint_ids + self.gripper_joint_ids

        # -----------------------------
        # finger body ids
        # joint와 마찬가지로 body 이름 → 내부 인덱스로 변환
        # 왼쪽/오른쪽 손가락 끝의 위치를 읽을 때 이 인덱스 사용
        # -----------------------------
        left_found = self._robot.find_bodies(self.cfg.left_finger_body_name)[0]
        if len(left_found) == 0:
            raise RuntimeError(f"왼쪽 finger body를 찾지 못함: {self.cfg.left_finger_body_name}")

        right_found = self._robot.find_bodies(self.cfg.right_finger_body_name)[0]
        if len(right_found) == 0:
            raise RuntimeError(f"오른쪽 finger body를 찾지 못함: {self.cfg.right_finger_body_name}")

        self.left_finger_body_id = int(left_found[0])
        self.right_finger_body_id = int(right_found[0])

        # 안쪽 관절 link (r1, l1) — outer fingertip(r2, l2)과 함께 그리퍼 수직 자세 정렬용
        # 두 중점 모두 박스 xy에 맞아야 그리퍼가 박스 위에 수직으로 위치
        r1_found = self._robot.find_bodies("rh_p12_rn_r1")[0]
        l1_found = self._robot.find_bodies("rh_p12_rn_l1")[0]
        if len(r1_found) == 0 or len(l1_found) == 0:
            raise RuntimeError("그리퍼 inner body 못 찾음: rh_p12_rn_r1 / rh_p12_rn_l1")
        self.right_inner_body_id = int(r1_found[0])
        self.left_inner_body_id = int(l1_found[0])

        # -----------------------------
        # 속도 스케일
        # 각 관절마다 action으로 얼마나 빠르게 움직일지 배율을 설정
        # ones_like → 모든 관절을 1.0으로 초기화 (= 그대로)
        # -----------------------------
        self.robot_dof_speed_scales = torch.ones_like(self.robot_dof_lower_limits)
        for gid in self.gripper_joint_ids:
            # 현재는 gripper도 1.0 (arm과 동일). 나중에 줄이려면 0.1 등으로 수정
            self.robot_dof_speed_scales[gid] = 1.0

        # USD에서 mimic 정보가 깨진 채 변환된 것으로 보임 → 코드에서 강제
        # URDF 사양: 4관절 모두 [0.0, 1.135] limit, 같은 위치를 따라가야 함
        # USD가 일부 limit을 다르게 가졌을 수 있으니 URDF spec으로 강제 통일
        urdf_grip_lower, urdf_grip_upper = 0.0, 1.135
        for gid in self.gripper_joint_ids:
            self.robot_dof_lower_limits[gid] = urdf_grip_lower
            self.robot_dof_upper_limits[gid] = urdf_grip_upper

        # -----------------------------
        # 실질적인 제어 주기 (초)
        # dt(물리 스텝) × decimation(몇 스텝마다 action) = 실제 action 주기
        # 예: 1/120 × 2 = 1/60초 → 60Hz로 제어
        # -----------------------------
        self.dt = self.cfg.sim.dt * self.cfg.decimation

        # -----------------------------
        # target / buffer
        # 매 스텝 계산 결과를 재사용하기 위해 미리 GPU 메모리에 텐서 할당
        # -----------------------------

        # 각 관절의 목표 위치 (position control 방식)
        # shape: (num_envs=256, num_joints=로봇 전체 관절 수)
        self.robot_dof_targets = torch.zeros(
            (self.num_envs, self._robot.num_joints),
            device=self.device,
        )

        # 물체의 월드 좌표 (x, y, z)
        # shape: (num_envs, 3)
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # 물체의 환경 원점 기준 상대 좌표 (observation에 넣는 값)
        # shape: (num_envs, 3)
        self.obj_pos_rel = torch.zeros((self.num_envs, 3), device=self.device)

        # 두 손가락 끝의 중간점 위치
        # shape: (num_envs, 3)
        self.grip_center_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # grasp target 위치 - grip center 사이의 벡터 (얼마나 틀어졌는지)
        # shape: (num_envs, 3)
        self.obj_to_grip = torch.zeros((self.num_envs, 3), device=self.device)

        # 그리퍼가 실제로 도달해야 할 목표점 위치 (물체 중심 + z offset)
        # shape: (num_envs, 3)
        self.grasp_target_pos = torch.zeros((self.num_envs, 3), device=self.device)

        # 이번 스텝의 action 저장용 (reward 계산 시 action penalty에 사용)
        # shape: (num_envs, action_space=7)
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 직전 스텝에서 그리퍼~target 거리 (접근 보상 계산 시 이전값과 비교)
        # shape: (num_envs,) → 환경 하나당 스칼라 1개
        self._prev_dist = torch.zeros(self.num_envs, device=self.device)

        # v47: grasp streak counter — 박스 잡힌 상태로 연속 step 유지 카운트
        # 조건 만족 시 +1, 깨지면 0 reset
        # cfg.grasp_hold_steps (60 step = 1초) 도달 시 success
        self.grasp_streak = torch.zeros(self.num_envs, device=self.device)

        # -----------------------------
        # reward log
        # train.py의 callback에서 reward_log를 읽어서 출력함
        # 딕셔너리 형태: {"보상항 이름": float값}
        # -----------------------------
        self.reward_log = { }

        # 초기화 직후에 중간값을 한 번 계산해서 버퍼를 채워둠
        self._compute_intermediate_values()

    # ------------------------------------------------------------------
    # Scene 설정
    # Isaac Lab이 환경 초기화 시 자동으로 호출하는 메서드
    # ------------------------------------------------------------------
    def _setup_scene(self):
        # 로봇을 씬에 추가 (Articulation = 관절이 있는 로봇)
        self._robot = Articulation(self.cfg.robot)

        # 물체(박스)를 씬에 추가 (RigidObject = 단순 강체)
        self._object = RigidObject(self.cfg.object)

        # 씬 매니저에 등록 — "robot", "object" 키로 나중에 접근 가능
        self.scene.articulations["robot"] = self._robot
        self.scene.rigid_objects["object"] = self._object

        # 지형 설정에 씬의 환경 수/간격을 복사 (지형이 씬 크기에 맞게 생성됨)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing

        # cfg에 지정된 terrain class(평면 등)를 인스턴스화해서 실제로 씬에 추가
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)

        # 환경 0번을 복사해서 num_envs개 환경을 만듦
        # copy_from_source=False → 소스 환경을 복사 원본으로 쓰지 않음
        self.scene.clone_environments(copy_from_source=False)

        # CPU 시뮬레이션일 때는 환경끼리 충돌 필터링 필요 (GPU는 자동 처리)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])

        # 반구형 조명 추가 (씬 전체를 균등하게 밝힘)
        # intensity=2000 → 밝기, color=(0.75, 0.75, 0.75) → 약간 회색빛 흰 조명
        light_cfg = sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        )
        # "/World/Light" 경로에 조명 오브젝트를 실제로 생성
        light_cfg.func("/World/Light", light_cfg)

    # ------------------------------------------------------------------
    # Action
    # ------------------------------------------------------------------
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # 네트워크 출력을 복사하고 -1~1 범위로 강제 제한
        # clamp = clip과 같은 의미, 범위 밖 값을 경계값으로 자름
        self.actions = actions.clone().clamp(-1.0, 1.0)

        # 새 목표 위치 = 현재 목표 + 속도스케일 × dt × action → 위치 제어 방식
        # robot_dof_targets: 현재 관절 목표 위치
        # robot_dof_speed_scales: 관절별 속도 배율 (현재 모두 1.0)
        # self.dt: 한 스텝의 시간 (1/60초)
        # actions_to_dof(self.actions): action 7개를 전체 관절 수 크기 텐서로 변환
        # cfg.action_scale: 전체 배율 (현재 1.0)
        targets = (
            self.robot_dof_targets
            + self.robot_dof_speed_scales
            * self.dt
            * self.actions_to_dof(self.actions)
            * self.cfg.action_scale
        )

        # 계산된 목표 위치를 관절 한계(lower~upper) 안으로 클램핑
        self.robot_dof_targets = torch.clamp(
            targets,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # Mimic 강제: master(rh_r1_joint)의 target을 mimic 관절에 복사
        # USD에서 mimic이 깨졌으므로 코드에서 동기화
        master_target = self.robot_dof_targets[:, self.main_gripper_joint_id:self.main_gripper_joint_id+1]
        for gid in self.mimic_gripper_joint_ids:
            self.robot_dof_targets[:, gid] = master_target.squeeze(-1)

    def actions_to_dof(self, actions: torch.Tensor) -> torch.Tensor:
        # action 7개를 로봇 전체 관절 수 크기의 텐서로 펼침
        # 제어하지 않는 관절은 0으로 채워짐 → 해당 관절은 움직이지 않음
        # shape: (num_envs, num_joints 전체)
        dof_delta = torch.zeros(
            (actions.shape[0], self._robot.num_joints),
            device=self.device,
        )

        # action의 0~5번 → arm joint 6개에 각각 할당
        # enumerate → (i=0, joint_id=실제인덱스), (i=1, ...) 순서로 반복
        # dof_delta[:, joint_id]: 모든 환경에서 해당 관절의 delta를 한번에 설정
        for i, joint_id in enumerate(self.arm_joint_ids):
            dof_delta[:, joint_id] = actions[:, i]

        # action의 6번 = gripper 명령, ×3 배율 적용 (속도를 더 크게)
        # actions[:, 6] → 256개 환경 전부의 gripper action을 한번에 가져옴
        grip_cmd = actions[:, 6] * 3.0 # 나중에 줄여야할듯

        # gripper joint 4개 전부에 동일한 명령을 할당 (mimic 구조 모사)
        for gid in self.gripper_joint_ids:
            dof_delta[:, gid] = grip_cmd

        return dof_delta

    def _apply_action(self) -> None:
        # 계산된 목표 위치를 물리 엔진에 실제로 전달
        self._robot.set_joint_position_target(self.robot_dof_targets)

        # ---- Mimic 강제: 매 step 직접 joint position 동기화 ----
        # OMY.usd에 mimic constraint가 충분히 표현 안 됨 (Isaac Sim importer 한계)
        # → master(rh_r1_joint) 의 실제 위치를 다른 3개 mimic 관절에 강제로 복사
        # 4-bar linkage 효과 모사 → fingertip 평행 유지
        master_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        master_vel = self._robot.data.joint_vel[:, self.main_gripper_joint_id]
        joint_pos = self._robot.data.joint_pos.clone()
        joint_vel = self._robot.data.joint_vel.clone()
        for gid in self.mimic_gripper_joint_ids:
            joint_pos[:, gid] = master_pos
            joint_vel[:, gid] = master_vel
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel)

    # ------------------------------------------------------------------
    # Observations
    # ------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 이번 스텝의 물체/그리퍼 위치 등 중간값을 최신으로 갱신
        self._compute_intermediate_values()

        # joint_pos: 현재 관절 위치, shape = (num_envs, len(all_obs_joint_ids))
        # all_obs_joint_ids = arm 6개 + gripper 4개 → 총 10개 관절만 뽑음
        joint_pos = self._robot.data.joint_pos[:, self.all_obs_joint_ids]
        joint_vel = self._robot.data.joint_vel[:, self.all_obs_joint_ids]

        # 해당 관절들의 하한/상한값
        lower = self.robot_dof_lower_limits[self.all_obs_joint_ids]
        upper = self.robot_dof_upper_limits[self.all_obs_joint_ids]

        # 관절 위치를 -1~1 범위로 정규화
        # (joint_pos - lower) / (upper - lower) → 0~1
        # × 2 - 1 → -1~1
        # 1e-8 → 분모가 0이 되는 걸 방지
        dof_pos_scaled = 2.0 * (joint_pos - lower) / (upper - lower + 1e-8) - 1.0

        # 관절 속도에 배율 적용 (현재 dof_velocity_scale=1.0이라 그대로)
        dof_vel_scaled = joint_vel * self.cfg.dof_velocity_scale

        # gripper 닫힘 상태 (0=완전히 열림, 1=완전히 닫힘)
        # main_gripper_joint_id 한 개만 읽음
        # [:, ...] → 모든 환경에서 해당 관절 위치를 가져옴
        # .unsqueeze(-1) → shape (num_envs,) → (num_envs, 1)로 만들어 cat에 맞춤
        grip_joint_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id].unsqueeze(-1)
        grip_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id].view(1, 1)
        grip_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id].view(1, 1)

        # gripper가 얼마나 닫혀 있는지 0~1로 정규화
        gripper_close_state = (grip_joint_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)

        # 목표 높이 - 물체의 현재 높이 = 얼마나 더 올려야 하는지
        # obj_pos_w[:, 2] → 모든 환경의 물체 z좌표 (높이)
        # unsqueeze(-1) → shape (num_envs,) → (num_envs, 1)
        to_lift_target = (self.cfg.lift_height_threshold - self.obj_pos_w[:, 2]).unsqueeze(-1)

        # 왼쪽/오른쪽 손가락 끝의 월드 좌표
        # body_pos_w shape: (num_envs, num_bodies, 3)
        # [:, left_finger_body_id, :] → 모든 환경에서 왼쪽 손가락 body의 (x,y,z)
        l_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]

        # 물체 위치 - 왼쪽 손가락 위치 = 손가락이 물체까지 어느 방향/얼마나 이동해야 하는지
        left_to_obj_vec  = self.obj_pos_w - l_pos
        right_to_obj_vec = self.obj_pos_w - r_pos

        # 34차원 observation 벡터 조립
        # dim=-1 → 마지막 차원(특징 차원)을 따라 이어 붙임
        # 결과 shape: (num_envs, 34)
        obs = torch.cat(
            [
                dof_pos_scaled,       # 7  (arm 6 + gripper 1 정규화 위치)
                dof_vel_scaled,       # 7  (arm 6 + gripper 1 속도)
                self.obj_pos_rel,     # 3  (물체의 환경 원점 기준 상대 위치)
                self.obj_to_grip,     # 3  (그리퍼 중심 → grasp target 벡터)
                left_to_obj_vec,      # 3  (왼손가락 → 물체 벡터)
                right_to_obj_vec,     # 3  (오른손가락 → 물체 벡터)
                gripper_close_state,  # 1  (0=열림, 1=닫힘)
                to_lift_target,       # 1  (목표 높이까지 남은 거리)
            ],
            dim=-1,
        )

        # -5~5로 클램핑 → 이상치가 네트워크에 과도한 영향을 미치는 것을 방지
        # "policy" 키로 반환 → DirectRLEnv 규격
        return {"policy": torch.clamp(obs, -5.0, 5.0)}

    # ------------------------------------------------------------------
    # Rewards
    # ------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        self._compute_intermediate_values()

        # --------------------------------------------------
        # 1) 기본 위치 정보 — 자주 쓰는 값을 짧은 이름으로 미리 꺼냄
        # --------------------------------------------------
        obj_pos = self.obj_pos_w          # 물체 월드 위치, shape (num_envs, 3)
        grip_pos = self.grip_center_pos   # 그리퍼 중심 월드 위치
        target_pos = self.grasp_target_pos  # 그리퍼가 도달해야 할 목표점

        # 왼쪽/오른쪽 손가락 끝 월드 위치 (r2, l2)
        l_pos = self._robot.data.body_pos_w[:, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[:, self.right_finger_body_id, :]

        # 안쪽 관절 link 월드 위치 (r1, l1) — 수직 자세 정렬용
        l_inner_pos = self._robot.data.body_pos_w[:, self.left_inner_body_id, :]
        r_inner_pos = self._robot.data.body_pos_w[:, self.right_inner_body_id, :]

        # gripper 닫힘 상태 (0~1)
        grip_joint_pos = self._robot.data.joint_pos[:, self.main_gripper_joint_id]
        grip_lower = self.robot_dof_lower_limits[self.main_gripper_joint_id]
        grip_upper = self.robot_dof_upper_limits[self.main_gripper_joint_id]

        # 0 = 완전히 열림, 1 = 완전히 닫힘
        gripper_close_state = (grip_joint_pos - grip_lower) / (grip_upper - grip_lower + 1e-8)

        # 그리퍼 중심 → grasp target 벡터
        target_to_grip = target_pos - grip_pos

        # 3D 거리 (유클리디안 노름)
        # dim=-1 → (x,y,z) 차원을 따라 합산 → 결과 shape (num_envs,)
        dist = torch.norm(target_to_grip, dim=-1)

        # XY 평면 거리 (수평 정렬 측정)
        # target_to_grip[:, :2] → x,y 성분만 (z 제외)
        xy_dist = torch.norm(target_to_grip[:, :2], dim=-1)

        # Z 거리 (높이 정렬 측정)
        # target_to_grip[:, 2] → z 성분 하나만 꺼냄
        z_dist = torch.abs(target_to_grip[:, 2])

        # --------------------------------------------------
        # 2) 접근 진행 보상
        # 이전 step보다 가까워졌을 때만 양수 보상
        # clamp(min=0.0) → 멀어졌을 때는 보상 0 (패널티 없음)
        # --------------------------------------------------
        approach_reward = torch.clamp(self._prev_dist - dist, min=0.0)

        # 이번 거리를 다음 스텝 비교를 위해 저장
        self._prev_dist = dist.clone()

        # --------------------------------------------------
        # 3) 정렬 보상
        # exp(-k * d^2) 형태 → 거리가 0에 가까울수록 1에 수렴, 멀수록 0에 수렴
        # k가 클수록 보상 범위가 좁아짐 (정밀도 요구가 높아짐)
        # --------------------------------------------------

        # XY 정렬 보상: k=40, xy_dist=0이면 1.0, 0.1m이면 ≈ 0.02
        xy_align_reward = torch.exp(-40.0 * xy_dist**2)

        # Z 정렬 보상: XY가 먼저 맞아야(xy_dist < 0.1) Z 보상도 활성화
        # Boolean 텐서를 float으로 쓰면 True=1.0, False=0.0
        z_align_reward = torch.exp(-60.0 * z_dist**2) * (xy_dist < 0.1)

        # 완전히 정렬됐다고 볼 임계값: XY, Z 모두 2cm 이내
        xy_aligned = xy_dist < 0.02
        z_aligned = z_dist < 0.02

        # & → 두 조건을 AND로 합침 (둘 다 True여야 True)
        aligned = xy_aligned & z_aligned

        # --------------------------------------------------
        # 4) 양옆 finger 배치 체크
        # 물체 y좌표 기준 왼손가락은 왼쪽(y가 작아야), 오른손가락은 오른쪽(y가 커야)
        # --------------------------------------------------

        # l_pos[:, 1] → 왼손가락의 y좌표. obj_pos[:, 1] → 물체의 y좌표
        # 왼손가락 y < 물체 y → 왼손가락이 물체 왼쪽에 있음 → True
        left_is_left = l_pos[:, 1] < obj_pos[:, 1]
        right_is_right = r_pos[:, 1] > obj_pos[:, 1]

        # 각 손가락과 물체 사이의 3D 거리
        left_to_obj = torch.norm(obj_pos - l_pos, dim=-1)
        right_to_obj = torch.norm(obj_pos - r_pos, dim=-1)

        # 두 손가락 모두 물체에서 8cm 이내에 있는지
        fingers_near = (left_to_obj < 0.08) & (right_to_obj < 0.08)
        side_ok = left_is_left & right_is_right

        # 두 손가락의 z좌표 차이 → 0에 가까울수록 수평으로 잘 정렬됨
        finger_z_diff = torch.abs(l_pos[:, 2] - r_pos[:, 2])

        # k=100 → 매우 좁은 범위에서만 높은 보상 (z 차이 0.01m=1cm 이하여야 높음)
        horizontal_reward = torch.exp(-100.0 * finger_z_diff**2)
        horizontal_ok = finger_z_diff < 0.01

        # aligned만으로 grasp 준비 판정
        # 그리퍼 중심이 목표점 2cm 이내면 물리적으로 잡을 수 있는 위치
        # fingers_near(=0.11)가 너무 까다로워서 is_grasping 자체가 안 뜨는 문제 해결
        pre_grasp_ready = aligned

        # --------------------------------------------------
        # 5) grasp 판정
        # --------------------------------------------------

        # 두 손가락 끝 사이의 거리 (벌어진 정도)
        finger_gap = torch.norm(l_pos - r_pos, dim=-1)

        # gripper가 65% 이상 닫혔으면 "충분히 닫혔다"고 판정
        closed_enough = gripper_close_state > 0.65

        # fingers_near 기반으로 grasping 판정
        # aligned 대신 fingers_near 사용:
        # 그리퍼가 닫히면 finger tip이 이동 → grip_center 변동 → aligned=False 되는 문제 해결
        # fingers_near(0.87)는 닫히는 중에도 안정적으로 True 유지됨
        is_grasping = fingers_near & closed_enough

        # -----------------------------
        # finger → object XY 거리 보상
        # 손가락이 물체 XY 위치에 가까울수록 높은 보상
        # -----------------------------

        # l_pos[:, :2] → 왼손가락의 x, y만. obj_pos[:, :2] → 물체의 x, y만
        left_xy_dist = torch.norm(l_pos[:, :2] - obj_pos[:, :2], dim=-1)
        right_xy_dist = torch.norm(r_pos[:, :2] - obj_pos[:, :2], dim=-1)

        # k=50 → xy_dist=0이면 1.0, 0.1m이면 ≈ 0.006
        left_xy_reward = torch.exp(-50.0 * left_xy_dist**2)
        right_xy_reward = torch.exp(-50.0 * right_xy_dist**2)

        # 왼쪽/오른쪽 보상 평균
        xy_close_reward = 0.5 * (left_xy_reward + right_xy_reward)

        # -----------------------------
        # finger → grasp target Z 거리 보상
        # 손가락이 "집어야 하는 높이"에 가까울수록 높은 보상
        # -----------------------------

        # grasp_target_pos[:, 2] → 목표 집기 높이 (물체 중심 + 5cm)
        target_z = self.grasp_target_pos[:, 2]

        left_z_dist = torch.abs(l_pos[:, 2] - target_z)
        right_z_dist = torch.abs(r_pos[:, 2] - target_z)

        # k=80 → z 정렬에 더 엄격한 기준
        left_z_reward = torch.exp(-80.0 * left_z_dist**2)
        right_z_reward = torch.exp(-80.0 * right_z_dist**2)

        z_close_reward = 0.5 * (left_z_reward + right_z_reward)

        # -----------------------------
        # 최종 close reward
        # XY, Z 정렬 + aligned + 그리퍼 닫힘 상태까지 곱함
        # gripper_close_state를 곱해야 "닫아야만" 높은 보상 → 안 닫으면 0
        # -----------------------------
        close_reward = xy_close_reward * z_close_reward * aligned.float() * gripper_close_state

        # grip center가 grasp target 8cm 이내일 때 gripper를 닫을수록 보상 (원본)
        near_object = (dist < 0.08).float()
        gripper_close_bonus = near_object * gripper_close_state * 5.0

        # 정렬 + 잡고있는 상태의 품질 점수 (현재 reward 식에는 직접 사용 안 됨)
        alignment_score = (
            torch.exp(-40.0 * xy_dist**2)
            * torch.exp(-60.0 * z_dist**2)
            * is_grasping.float()
        )

        # --------------------------------------------------
        # 6) lift 보상
        # is_grasping=True일 때만 들어올린 높이에 비례해서 보상
        # --------------------------------------------------

        # obj_pos[:, 2] → 모든 환경의 물체 현재 높이 (z좌표)
        obj_height = obj_pos[:, 2]

        # cfg에서 설정한 물체 초기 z 위치 (0.06m)
        # init_state.pos = (0.45, -0.10, 0.06) 에서 [2]번째 = z
        base_object_height = self.cfg.object.init_state.pos[2]

        # v22: finger_proximity를 max-based로 변경 (양쪽 손가락 모두 박스에 닿아야)
        # v21 평균 거리는 한쪽만 닿으면 보상 → 정렬과 충돌. max는 더 먼 손가락 기준
        # max_dist=5cm→0.29, 3cm→0.64, 1cm→0.95 (exp -50)
        max_finger_dist = torch.maximum(left_to_obj, right_to_obj)
        finger_proximity = torch.exp(-50.0 * max_finger_dist ** 2)
        closed_smooth_for_lift = torch.clamp(gripper_close_state, 0.0, 1.0)
        lift_gate = finger_proximity * closed_smooth_for_lift   # 0~1 연속

        # v26: lift_progress denominator 0.14 → 0.06 (6cm 들기에서 1.0, success threshold 0.12와 일치)
        # curriculum 1단계: 박스 6cm 들면 success → 첫 success 유발
        # 60% 달성 후 14cm로 복원
        lift_progress = torch.clamp(
            (obj_height - base_object_height) / 0.06,
            min=0.0,
            max=1.0,
        )
        # v25: ^2 shaping 폐기, linear 복원
        # 이유: v24에서 ^2가 낮은 lift 영역(3cm) 보상 5배 약화 → lift 자체 학습 후퇴
        # linear가 partial lift도 비례 보상해서 학습 시작에 더 좋음
        lift_reward = lift_progress * lift_gate

        # v28: high_lift_bonus 추가 (squeeze-induced 1.7cm 공짜 lift 차단)
        # 가설: 그리퍼 닫는 동작 자체가 박스 1.7cm 올림 → 정책이 그 보상에 안주
        # 해결: 박스 3cm 이상에서만 추가 보상 → 실제 팔로 들기 강제
        # 3cm: 0, 4.5cm: 0.5, 6cm: 1.0
        lift_above_3cm = torch.clamp(
            (obj_height - base_object_height - 0.03) / 0.03,
            min=0.0,
            max=1.0,
        )
        high_lift_bonus = lift_above_3cm * lift_gate

        # --------------------------------------------------
        # 6-1) 물체가 넘어졌는지 판정
        # 쿼터니언으로 물체의 "위 방향"이 실제로 위를 향하는지 확인
        # --------------------------------------------------

        # 물체의 회전 쿼터니언 (w, x, y, z 형식)
        # shape: (num_envs, 4)
        obj_quat = self._object.data.root_quat_w

        # 물체 로컬 좌표계에서의 "위쪽" 단위벡터 (0, 0, 1)
        # shape: (num_envs, 3) → 모두 0으로 초기화 후 z만 1로 설정
        local_up = torch.zeros((obj_quat.shape[0], 3), device=self.device)
        local_up[:, 2] = 1.0

        # 쿼터니언으로 로컬 up 벡터를 월드 좌표계로 변환
        # 결과: 물체의 위쪽 방향이 월드에서 어느 방향을 가리키는지
        world_up_from_obj = quat_apply(obj_quat, local_up)

        # world_up_from_obj[:, 2] → 월드 z방향 성분 (위를 향할수록 1에 가까움)
        # 물체가 똑바로 서있으면 ≈ 1.0, 90도 넘어지면 ≈ 0, 뒤집히면 ≈ -1
        upright_score = world_up_from_obj[:, 2]

        # upright_score < 0.3 → 약 72도 이상 기울어진 상태 → 넘어진 것으로 판단
        fallen = upright_score < 0.1

        # --------------------------------------------------
        # 7) 성공 보상
        # --------------------------------------------------

        # 물체 높이 > 0.2m이면 성공 (True → 1.0, False → 0.0)
        success = (obj_height > self.cfg.lift_height_threshold).float()
        success_reward = success  # 현재 최종 reward 식에서 주석처리됨

        # --------------------------------------------------
        # 8) 액션 패널티
        # 불필요하게 큰 action을 줄이도록 유도
        # sum(actions^2) → 7개 action의 제곱합 = L2 norm의 제곱
        # --------------------------------------------------
        action_penalty = torch.sum(self.actions ** 2, dim=-1)

        # --------------------------------------------------
        # 9) 최종 reward 합산
        # 각 항의 가중치는 중요도에 따라 튜닝된 값
        # --------------------------------------------------

        # 물체 근처(8cm 이내)가 아닌데 그리퍼를 닫으면 패널티
        # near_object=0(멀다) × gripper_close_state=1(닫힘) → 최대 -5.0
        # near_object=1(가깝다) → 패널티 0 (자유롭게 닫아도 됨)
        premature_close_penalty = (1.0 - near_object) * gripper_close_state

        # 물체 근처(8cm 이내)일 때 그리퍼를 닫으면 보상
        # premature_close_penalty의 반대 신호 — "가까우면 닫아라"
        grasp_close_reward = near_object * gripper_close_state

        # 진단용: grip-obj 중심 거리 로깅 (보상 항목으로는 미사용)
        grip_to_obj_center = torch.norm(self.grip_center_pos - self.obj_pos_w, dim=-1)

        # 안쪽 관절 (r1, l1) 중점도 박스 xy 정렬 (수직 자세 강제, 약하게)
        # outer fingertip 중점(r2,l2)과 inner joint 중점(r1,l1) 둘 다 박스 위 같은 xy
        # → 그리퍼 축이 박스 위에 수직 위치 (비스듬히 잡기 차단)
        inner_mid_xy = 0.5 * (l_inner_pos[:, :2] + r_inner_pos[:, :2])
        inner_xy_dist = torch.norm(inner_mid_xy - obj_pos[:, :2], dim=-1)
        inner_xy_align_reward = torch.exp(-40.0 * inner_xy_dist ** 2)

        # 박스가 그리퍼 내부에 있는지 = outer xy + inner xy + z 모두 정렬 (v15 복원)
        # fingers_near 곱셈은 학습 신호 0으로 만들어 학습 불가 → 제거
        box_inside_gripper = xy_align_reward * inner_xy_align_reward * z_align_reward

        # 그리퍼 닫힘 정도 (continuous 0~1) — 닫는 중에도 점진적 보상
        closed_smooth = torch.clamp(gripper_close_state, 0.0, 1.0)

        # v24: 가짜 잡기 (false grip) 차단
        # v22 15M 시각화 결과: 박스 옆에서 그리퍼 닫고 hovering하는 local optimum 발견
        # 원인: tight_grip_close/squeeze가 박스를 실제로 들지 않아도 보상됨
        # 변경: 박스가 실제로 들렸을 때 (5mm 이상)만 close/squeeze 보상
        # 또한 가짜 잡기에 직접 페널티
        box_lifted_5mm = (obj_height > base_object_height + 0.005).float()             # 박스 5mm 이상 떴는지
        box_lifted_smooth = torch.clamp((obj_height - base_object_height) / 0.02, 0.0, 1.0)  # 0~1, 2cm에서 1.0

        # v30: side_ok 폐기 (너무 엄격 → 학습 붕괴)
        # 대체: in_grasp = 손가락 끝 중점이 박스 중심 가까이 (3cm 이내)
        # side_ok와 달리 박스를 어떤 방향으로 감싸도 만족 → 부드러운 신호
        fingertips_mid = 0.5 * (l_pos + r_pos)
        mid_to_box_dist = torch.norm(fingertips_mid - obj_pos, dim=-1)
        # 손가락 중점이 박스 중심 3cm 이내면 박스가 그리퍼 안에 있음
        in_grasp = torch.exp(-200.0 * mid_to_box_dist ** 2)  # 1cm: 0.98, 3cm: 0.16, 5cm: 0.007

        tight_grip_align = box_inside_gripper                                          # 0~1: 정렬 (조건 무관)
        # close/squeeze: 박스 들렸을 때 + 박스가 그리퍼 안에 있을 때만
        tight_grip_close = closed_smooth * near_object * box_lifted_smooth * in_grasp
        tight_grip_squeeze = torch.exp(-50.0 * finger_gap ** 2) * near_object * box_lifted_smooth * in_grasp
        tight_grip_reward = tight_grip_align + tight_grip_close + tight_grip_squeeze   # 0~3 합

        # v24: 가짜 잡기 페널티 (closed near box without lift)
        false_grip_penalty = (closed_smooth > 0.7).float() * (1.0 - box_lifted_5mm) * near_object

        # v30: pushing penalty — 그리퍼 닫고 박스 근처에 있는데 박스가 그리퍼 안에 없으면 페널티
        # 그리퍼가 박스 옆에서 미는 동작 차단 (in_grasp < 0.5 = 손가락 중점이 박스에서 멀음)
        pushing_penalty = (closed_smooth > 0.5).float() * near_object * (1.0 - in_grasp)

        # v31: opened_when_far — 박스에서 멀 때(8cm 이상) 그리퍼 열어둔 상태 보상
        # 정책이 박스 접근 중에 그리퍼 닫는 자세를 학습하지 못하도록
        opened_smooth = 1.0 - closed_smooth
        not_near_object = 1.0 - near_object
        opened_when_far = opened_smooth * not_near_object

        # v30: lift_gate에 in_grasp 적용 — 박스가 그리퍼 안에 있을 때만 lift_reward
        # 미는 동작으로 박스 떠도 in_grasp 작아서 lift_reward 못 받음
        lift_gate = lift_gate * in_grasp
        lift_reward = lift_progress * lift_gate
        high_lift_bonus = lift_above_3cm * lift_gate

        # v44: 조건부 lift — in_grasp > 0.7 (진짜 잡기)일 때만 lift 보상
        # v43에서 lift weight 200이 grip 망가뜨림 (aligned 0.66→0.008)
        # 조건부 lift: grip 유지하면서 진짜 잡기 시 lift 학습 유도
        upright_smooth = (upright_score + 1.0) * 0.5

        centered_grip = in_grasp * closed_smooth_for_lift * upright_smooth
        close_inside_bonus = closed_smooth_for_lift * in_grasp
        gap_when_grasping = torch.clamp(finger_gap - 0.04, min=0.0) * in_grasp

        # v47: lift 완전 제거, grasp + N step hold task
        # v47b: closed_smooth > 0.5 제거 (박스 접촉으로 도달 불가) → finger_gap < 0.10 사용
        # 박스 두께 4.4cm + 여유 1.5cm. 박스 양옆에 손가락 닿은 상태 = grasp
        grasping_now = (
            (in_grasp > 0.5)                  # 박스가 손가락 중점에
            & (finger_gap < 0.10)             # 손가락이 박스 양옆에 닿음 (4.4 + 1.5cm 여유)
            & (upright_smooth > 0.95)         # 박스 거의 정자세
        )
        # streak update (in-place 안 됨, where로 새 텐서)
        new_streak = torch.where(grasping_now, self.grasp_streak + 1.0, torch.zeros_like(self.grasp_streak))
        self.grasp_streak = new_streak
        streak_progress = torch.clamp(self.grasp_streak / float(self.cfg.grasp_hold_steps), 0.0, 1.0)
        streak_success = (self.grasp_streak >= float(self.cfg.grasp_hold_steps)).float()

        reward = (
            + 30.0 * finger_proximity
            + 50.0 * in_grasp
            + 50.0 * centered_grip
            + 50.0 * close_inside_bonus
            + 10.0 * upright_smooth
            + 10.0 * opened_when_far
            + 30.0 * streak_progress          # v47: streak ramp 0~1
            + 1000.0 * streak_success         # v47: N step 유지 success
            - 50.0 * gap_when_grasping
            - 0.001 * action_penalty
        )

        # --------------------------------------------------
        # 10) 로그 — train.py의 callback이 읽어서 화면에 출력함
        # float(tensor.mean()) → 256개 환경의 평균값을 파이썬 float으로 변환
        # --------------------------------------------------
        self.reward_log["approach_reward"] = float(approach_reward.mean())
        self.reward_log["xy_align_reward"] = float(xy_align_reward.mean())
        self.reward_log["inner_xy_align"] = float(inner_xy_align_reward.mean())
        self.reward_log["tight_grip"] = float(tight_grip_reward.mean())
        self.reward_log["z_align_reward"] = float(z_align_reward.mean())
        self.reward_log["lift_reward"] = float(lift_reward.mean())
        self.reward_log["close_reward"] = float(close_reward.mean())
        self.reward_log["horizontal_reward"] = float(horizontal_reward.mean())  # 진단용 (보상 미사용)
        self.reward_log["finger_gap"] = float(finger_gap.mean())
        self.reward_log["aligned"] = float(aligned.float().mean())
        self.reward_log["fingers_near"] = float(fingers_near.float().mean())
        self.reward_log["pre_grasp_ready"] = float(pre_grasp_ready.float().mean())
        self.reward_log["premature_close"] = float(premature_close_penalty.mean())
        self.reward_log["grasp_close_reward"] = float(grasp_close_reward.mean())
        self.reward_log["near_object"] = float(near_object.mean())
        self.reward_log["success_reward"] = float(success_reward.mean())
        self.reward_log["grip_to_obj_center"] = float(grip_to_obj_center.mean())
        self.reward_log["finger_proximity"] = float(finger_proximity.mean())
        self.reward_log["lift_gate"] = float(lift_gate.mean())
        self.reward_log["false_grip_penalty"] = float(false_grip_penalty.mean())
        self.reward_log["box_lifted_smooth"] = float(box_lifted_smooth.mean())
        self.reward_log["high_lift_bonus"] = float(high_lift_bonus.mean())
        self.reward_log["pushing_penalty"] = float(pushing_penalty.mean())
        self.reward_log["in_grasp"] = float(in_grasp.mean())
        self.reward_log["opened_when_far"] = float(opened_when_far.mean())
        self.reward_log["grasp_streak_avg"] = float(self.grasp_streak.mean())
        self.reward_log["streak_progress"] = float(streak_progress.mean())
        self.reward_log["streak_success_rate"] = float(streak_success.mean())
        self.reward_log["grasping_now_rate"] = float(grasping_now.float().mean())


        return reward

    # ========================================================= #
    # (이전 버전 reward 함수 — 참고용으로 주석 처리됨)
    # ========================================================= #

    # ------------------------------------------------------------------
    # Dones
    # 에피소드를 끝낼지 말지 결정
    # 반환: (terminated, truncated)
    #   terminated = 태스크 조건으로 끝남 (성공/실패)
    #   truncated  = 시간 제한으로 끝남
    # ------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self._compute_intermediate_values()

        # 물체의 현재 높이
        obj_height = self.obj_pos_w[:, 2]

        # 물체 넘어짐 판정 (reward 계산과 동일한 방법)
        obj_quat = self._object.data.root_quat_w

        local_up = torch.zeros((obj_quat.shape[0], 3), device=self.device)
        local_up[:, 2] = 1.0

        world_up_from_obj = quat_apply(obj_quat, local_up)
        upright_score = world_up_from_obj[:, 2]

        # 기울어짐 reset: 박스가 바닥 근처에 있을 때만 (들어올리는 도중 회전은 허용)
        # obj_height < 0.07 = 박스 중심이 바닥 근처 (박스 z=0.118이라 중심 0.07 미만이면 거의 누움)
        on_ground = obj_height < 0.07
        fallen_on_ground = (upright_score < 0.1) & on_ground

        # v47: success 기준 변경 — lift threshold 제거, streak 기반
        # streak >= grasp_hold_steps (60) 도달하면 success terminate
        streak_done = self.grasp_streak >= float(self.cfg.grasp_hold_steps)

        # 종료 조건 (하나라도 True면 에피소드 종료)
        terminated = (
            streak_done                                      # v47: N step 유지 → success
            | (obj_height < -0.1)                            # 물체가 바닥 아래로 추락
            | fallen_on_ground                               # 바닥에서만 기울어짐 reset
        )

        # episode_length_buf: 각 환경이 현재 에피소드에서 몇 스텝 진행했는지
        # max_episode_length - 1 이상이면 시간 초과 (truncated)
        truncated = self.episode_length_buf >= self.max_episode_length - 1

        return terminated, truncated

    # ------------------------------------------------------------------
    # Reset
    # 에피소드가 끝난 환경들만 골라서 초기화
    # env_ids: 초기화할 환경 번호 목록
    # ------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # None이면 전체 환경 초기화 (맨 처음 시작할 때)
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 부모 클래스에서 episode_length_buf 등 공통 카운터 초기화
        super()._reset_idx(env_ids)

        # env_ids를 GPU 텐서로 변환 (텐서 인덱싱에 사용)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # 이전 거리 버퍼를 0으로 초기화 (approach_reward 계산을 새로 시작)
        self._prev_dist[env_ids_t] = 0.0

        # v47: grasp streak 초기화 (새 에피소드 시작)
        self.grasp_streak[env_ids_t] = 0.0

        # -----------------------------
        # 1) 로봇 reset
        # -----------------------------

        # default_joint_pos: URDF/USD에 정의된 기본 자세의 관절 위치
        # [env_ids_t] → 초기화할 환경들의 기본 자세만 가져옴
        joint_pos = self._robot.data.default_joint_pos[env_ids_t].clone()

        # 기본 자세가 관절 한계를 벗어날 수 있으므로 클램핑
        joint_pos = torch.clamp(
            joint_pos,
            self.robot_dof_lower_limits,
            self.robot_dof_upper_limits,
        )

        # Mimic: gripper 4관절 모두 master 위치로 통일 (초기 상태도 대칭)
        master_init = joint_pos[:, self.main_gripper_joint_id:self.main_gripper_joint_id+1].clone()
        for gid in self.gripper_joint_ids:
            joint_pos[:, gid] = master_init.squeeze(-1)

        # 초기 속도는 0 (정지 상태에서 시작)
        joint_vel = torch.zeros_like(joint_pos)

        # 목표 위치도 초기 자세로 맞춤 (이전 에피소드의 목표가 남아있지 않도록)
        self.robot_dof_targets[env_ids_t] = joint_pos

        # 물리 엔진에 목표 위치 전달 (컨트롤러가 이 자세를 추종하게)
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids_t)

        # 시뮬레이션 내부 관절 상태 직접 덮어씌움 (텔레포트처럼 즉시 이동)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids_t)

        # -----------------------------
        # 2) 물체 reset
        # -----------------------------

        # 물체의 기본 상태 (위치, 회전, 속도) 가져옴
        # shape: (len(env_ids), 13) → [0:3]=pos, [3:7]=quat, [7:10]=linvel, [10:13]=angvel
        obj_state = self._object.data.default_root_state[env_ids_t].clone()

        # XY 위치 노이즈 생성
        # torch.rand → 0~1 균일분포. -0.5해서 -0.5~0.5. ×2해서 -1~1. ×noise로 범위 조절
        # shape: (len(env_ids), 2) → x, y 두 개
        noise = (
            (torch.rand((len(env_ids_t), 2), device=self.device) - 0.5)
            * 2.0
            * self.cfg.object_pos_noise
        )

        # 각 환경의 원점 위치 (환경마다 다른 오프셋)
        env_origins = self.scene.env_origins[env_ids_t]

        # cfg에서 초기 위치 분리 (base_x=0.45, base_y=-0.10, base_z=0.06)
        base_x, base_y, base_z = self.cfg.object.init_state.pos

        # 환경 원점 + 기본 오프셋 + 랜덤 노이즈 = 최종 물체 초기 위치
        # obj_state[:, 0] → 모든 초기화 환경의 x좌표
        obj_state[:, 0] = env_origins[:, 0] + base_x + noise[:, 0]
        obj_state[:, 1] = env_origins[:, 1] + base_y + noise[:, 1]
        obj_state[:, 2] = env_origins[:, 2] + base_z  # z는 노이즈 없음 (항상 동일 높이)

        # obj_state[:, 7:] → 인덱스 7 이후 = 선속도(7:10) + 각속도(10:13)를 모두 0으로
        obj_state[:, 7:] = 0.0

        # 물리 엔진에 물체 상태 직접 덮어씌움
        self._object.write_root_state_to_sim(obj_state, env_ids=env_ids_t)

        # reset된 환경들의 중간값 갱신
        self._compute_intermediate_values(env_ids_t)

    # ------------------------------------------------------------------
    # 중간값 계산
    # observation, reward, done에서 공통으로 사용하는 값들을 미리 계산해서 저장
    # env_ids가 None이면 전체 환경, 아니면 지정된 환경만 업데이트
    # ------------------------------------------------------------------
    def _compute_intermediate_values(self, env_ids: torch.Tensor | None = None):
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 왼쪽/오른쪽 손가락 끝의 월드 위치
        # body_pos_w[env_ids, body_id, :] → 특정 환경들의 특정 body 위치 (x,y,z)
        l_pos = self._robot.data.body_pos_w[env_ids, self.left_finger_body_id, :]
        r_pos = self._robot.data.body_pos_w[env_ids, self.right_finger_body_id, :]

        # 그리퍼 중심 = 두 손가락 끝의 평균 위치
        self.grip_center_pos[env_ids] = 0.5 * (l_pos + r_pos)

        # 물체 중심의 월드 좌표
        # root_pos_w → 강체의 루트(중심) 위치
        self.obj_pos_w[env_ids] = self._object.data.root_pos_w[env_ids]

        # 월드 좌표에서 환경 원점을 빼면 → 환경 내 상대 좌표
        # 이걸 obs에 넣어야 멀티 환경에서도 로봇-물체 관계가 동일하게 보임
        self.obj_pos_rel[env_ids] = self.obj_pos_w[env_ids] - self.scene.env_origins[env_ids]

        # grasp target = 물체 중심 위치에서 z를 5cm 올린 위치
        # 박스 중심보다 살짝 위를 잡으면 더 안정적으로 파지 가능
        self.grasp_target_pos[env_ids] = self.obj_pos_w[env_ids].clone()
        self.grasp_target_pos[env_ids, 2] += self.cfg.grasp_target_z_offset  # z += 0.05

        # grasp target → grip center 벡터 (얼마나, 어느 방향으로 틀어져 있는지)
        # observation에 넣어서 네트워크가 방향을 학습하도록 함
        self.obj_to_grip[env_ids] = self.grasp_target_pos[env_ids] - self.grip_center_pos[env_ids]
