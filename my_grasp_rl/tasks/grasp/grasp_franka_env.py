# __future__ 에서 annotations를 가져온다.
# 이건 타입 힌트(type hint)를 더 유연하게 쓰게 해주는 기능이다.
# 예를 들어 아직 정의되지 않은 클래스 이름을 타입으로 써도 처리하기 좋게 해준다.
from __future__ import annotations

# collections.abc 모듈의 Sequence 타입을 가져온다.
# Sequence는 리스트(list), 튜플(tuple)처럼 "순서가 있는 자료형"의 공통 타입으로 자주 쓴다.
from collections.abc import Sequence

# PyTorch 라이브러리 import
# tensor 계산, GPU 계산 등에 사용한다.
import torch

# isaaclab.sim 모듈을 sim_utils 라는 별명(alias)으로 import
# "import A as B" 문법은 A를 B라는 짧은 이름으로 부르겠다는 뜻이다.
import isaaclab.sim as sim_utils

# isaaclab.assets 모듈에서 Articulation, RigidObject 클래스만 가져온다.
# from A import B, C 문법은 A 안에서 필요한 것만 꺼내 쓰는 방식이다.
from isaaclab.assets import Articulation, RigidObject

# DirectRLEnv 클래스 import
# 이 클래스를 상속받아 직접 RL 환경을 만든다.
from isaaclab.envs import DirectRLEnv

# 장면(scene) 관련 클래스 import
from isaaclab.scene import InteractiveScene

# 파일 기반 스포너에서 바닥 생성 함수 import
from isaaclab.sim.spawners.from_files import spawn_ground_plane

# 랜덤값 샘플링 함수 import
from isaaclab.utils.math import sample_uniform

# 현재 폴더(.) 안의 grasp_franka_env_cfg 파일에서 설정 클래스 import
# . 는 "현재 패키지 위치"를 의미한다.
from .grasp_franka_env_cfg import GraspFrankaEnvCfg


# class 클래스이름(부모클래스):
# 라는 문법은 "부모 클래스를 상속받는 새로운 클래스 정의"이다.
# 여기서는 DirectRLEnv를 상속받아 GraspFrankaEnv를 만든다.
class GraspFrankaEnv(DirectRLEnv):
    """Franka grasp + lift Direct RL environment.

    Grasp RL (SAC/PPO) 기준 핵심 구성:
    - State  : robot joint state + gripper state + end-effector pos + object pos + relative vector
    - Action : continuous joint delta + gripper open/close scalar
    - Reward : reach + grasp + lift - penalties
    """

    # 타입 힌트
    # cfg 라는 멤버변수는 GraspFrankaEnvCfg 타입이라고 알려주는 용도다.
    # 실제로 값을 넣는 코드는 __init__ 또는 부모 클래스 쪽에서 처리된다.
    cfg: GraspFrankaEnvCfg

    # 생성자(constructor)
    # 객체를 만들 때 자동으로 호출된다.
    # self : 자기 자신 객체
    # cfg : 환경 설정 객체
    # render_mode : 렌더링 모드, 문자열 또는 None
    # **kwargs : 추가 키워드 인자를 모두 받겠다는 뜻
    def __init__(self, cfg: GraspFrankaEnvCfg, render_mode: str | None = None, **kwargs):
        # super()는 부모 클래스에 접근할 때 사용
        # 부모 클래스의 __init__ 을 먼저 실행해서 기본 환경 초기화를 한다.
        super().__init__(cfg, render_mode, **kwargs)

        # -----------------------------
        # joint / body indices
        # -----------------------------

        # 리스트 컴프리헨션 문법:
        # [식 for 변수 in 반복대상]
        # self.cfg.arm_joint_names 안에 있는 각 joint 이름(name)에 대해
        # self._robot.find_joints(name)로 조인트를 찾고,
        # 반환값 중 [0][0] 위치의 인덱스를 꺼내 리스트로 만든다.
        self.arm_joint_ids = [self._robot.find_joints(name)[0][0] for name in self.cfg.arm_joint_names]

        # 왼쪽 손가락 조인트의 인덱스를 찾는다.
        self.left_finger_id = self._robot.find_joints(self.cfg.left_finger_joint_name)[0][0]

        # 오른쪽 손가락 조인트의 인덱스를 찾는다.
        self.right_finger_id = self._robot.find_joints(self.cfg.right_finger_joint_name)[0][0]

        # end-effector body의 인덱스를 찾는다.
        # end-effector = 로봇팔 끝부분
        self.ee_body_id = self._robot.find_bodies(self.cfg.ee_body_name)[0][0]

        # len() 함수는 길이를 구한다.
        # 여기서는 arm joint 개수
        self.num_arm_dofs = len(self.arm_joint_ids)

        # -----------------------------
        # buffers
        # -----------------------------

        # torch.zeros((행, 열), device=...)
        # -> 0으로 채워진 텐서를 생성
        # self.num_envs 개의 환경 각각에 대해 action_space 크기의 action 저장 버퍼 생성
        self.actions = torch.zeros((self.num_envs, self.cfg.action_space), device=self.device)

        # 팔 관절용 action 버퍼
        self.arm_action = torch.zeros((self.num_envs, self.num_arm_dofs), device=self.device)

        # gripper용 action 버퍼 (1차원)
        self.gripper_action = torch.zeros((self.num_envs, 1), device=self.device)

        # 기본 joint position을 복사(clone)해서 joint target으로 사용
        # clone()은 원본과 별도의 복사본을 만든다.
        self.joint_targets = self._robot.data.default_joint_pos.clone()

        # end-effector world position 저장 버퍼
        self.ee_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # object world position 저장 버퍼
        self.obj_pos_w = torch.zeros((self.num_envs, 3), device=self.device)

        # object - ee 상대 벡터 저장 버퍼
        self.obj_to_ee = torch.zeros((self.num_envs, 3), device=self.device)

        # grasp 여부 저장 버퍼
        # dtype=torch.bool -> True/False 저장용
        self.grasped = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)

    # ---------------------------------------------------------------------
    # scene setup
    # ---------------------------------------------------------------------

    # 메서드 정의
    # self를 첫 번째 인자로 받는다.
    """
    환경 안에 무엇을 놓을지 정하는 함수.
    로봇, 물체, 바닥, 조명, 여러 env복제
    """
    def _setup_scene(self):
        # Articulation: 관절(joint)이 있는 로봇 같은 객체
        # cfg.robot_cfg 설정으로 로봇 생성
        self._robot = Articulation(self.cfg.robot_cfg)

        # RigidObject: 강체 물체
        # cfg.object_cfg 설정으로 물체 생성
        self._object = RigidObject(self.cfg.object_cfg)

        # InteractiveScene 생성
        self.scene = InteractiveScene(self.cfg.scene)

        # scene 안의 articulations 딕셔너리에 robot 등록
        self.scene.articulations["robot"] = self._robot

        # scene 안의 rigid_objects 딕셔너리에 object 등록
        self.scene.rigid_objects["object"] = self._object

        # 바닥 생성
        # prim_path="/World/ground" -> USD 월드 상의 경로
        # cfg=self.cfg.ground -> 바닥 설정
        spawn_ground_plane(prim_path="/World/ground", cfg=self.cfg.ground)

        # 여러 환경(env)을 복제
        # copy_from_source=False 는 원본 복사 방식 관련 옵션
        self.scene.clone_environments(copy_from_source=False)

        # 충돌 필터 설정
        # global_prim_paths=[] -> 특별히 제외할 글로벌 prim 없음
        self.scene.filter_collisions(global_prim_paths=[])

        # 조명 설정 객체 생성
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))

        # func(...) 호출로 실제 라이트를 월드에 생성
        light_cfg.func("/World/Light", light_cfg)

    # ---------------------------------------------------------------------
    # action
    # ---------------------------------------------------------------------

    # physics step 전에 action을 받아 내부 버퍼에 저장
    # actions: torch.Tensor 타입
    # -> None : 반환값이 없다는 뜻
    """
        정책이 낸 action을 받아서 내부적으로 정리하는 단계
        전체 action 저장
        팔 ACTION 분리
        gripper action 분리
    """
    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        # actions.clone()으로 action 복사 저장
        self.actions = actions.clone()

        # 모든 환경(:,)에 대해 앞 num_arm_dofs 개를 arm_action으로 사용
        # [:, : self.num_arm_dofs] 문법:
        # 모든 행, 0번째 열부터 num_arm_dofs 직전까지
        self.arm_action = actions[:, : self.num_arm_dofs]

        # 마지막 1개 값을 gripper action으로 사용
        # [-1:] 는 마지막 원소부터 끝까지
        self.gripper_action = actions[:, -1:].clone()

    # 실제 action을 로봇에 적용하는 함수
    """
        전처리된 action을 실제로 로봇에 적용하는 단계
    """
    def _apply_action(self) -> None:
        # 현재 arm joint target만 추출
        current_targets = self.joint_targets[:, self.arm_joint_ids]

        # 새 목표값 = 현재값 + 스케일 * action
        # continuous joint-delta control
        new_targets = current_targets + self.cfg.arm_action_scale * self.arm_action

        # 계산한 새 target을 전체 joint_targets의 arm joint 위치에 반영
        self.joint_targets[:, self.arm_joint_ids] = new_targets

        # ==== #
        current_left = self.joint_targets[:, self.left_finger_id]
        current_right = self.joint_targets[:, self.right_finger_id]

        delta = self.cfg.gripper_action_scale * self.gripper_action.squeeze(-1)

        new_left = current_left - delta
        new_right = current_right - delta

        new_left = torch.clamp(
            new_left,
            min=self.cfg.gripper_close_target,
            max=self.cfg.gripper_open_target,
        )
        new_right = torch.clamp(
            new_right,
            min=self.cfg.gripper_close_target,
            max=self.cfg.gripper_open_target,
        )

        self.joint_targets[:, self.left_finger_id] = new_left
        self.joint_targets[:, self.right_finger_id] = new_right
        # ==== #
        # squeeze(-1):
        # 마지막 차원이 크기 1이면 제거
        # 예: (N,1) -> (N,)
        # gripper action > 0 이면 닫기
        # close_mask = self.gripper_action.squeeze(-1) > 0.0

        # # ~ 는 bool 반전(not)
        # # 닫기가 아니면 열기
        # open_mask = ~close_mask

        # # close_mask가 True인 환경들만 선택해서 왼손가락 joint target 설정
        # self.joint_targets[close_mask, self.left_finger_id] = self.cfg.gripper_close_target

        # # 오른손가락도 닫기 target 설정
        # self.joint_targets[close_mask, self.right_finger_id] = self.cfg.gripper_close_target

        # # open_mask가 True인 환경들은 열기 target 설정
        # self.joint_targets[open_mask, self.left_finger_id] = self.cfg.gripper_open_target
        # self.joint_targets[open_mask, self.right_finger_id] = self.cfg.gripper_open_target

        # 최종 joint target을 로봇에 전달
        self._robot.set_joint_position_target(self.joint_targets)

    # ---------------------------------------------------------------------
    #
    """ 
    정책이 다음 action을 결정할 때 쓸 상태 벡터를 만드는 함수
    observations 관측 값 계산
        joint_pos : 팔 관절 위치
        joint_vel : 팔 관절 속도
        finger_pos : 손가락 위치
        finger_vel : 손가락 속도
        ee_pos_w : end-effector 위치
        obj_pos_w : 물체 위치
        obj_to_ee : 물체와 손끝 사이 상대벡터
        obj_lin_vel : 물체 속도
    """
    # State = joint_pos + joint_vel + gripper_pos + gripper_vel + ee_pos + obj_pos + obj_to_ee + obj_lin_vel
    # total = 7 + 7 + 2 + 2 + 3 + 3 + 3 + 3 = 30
    # ---------------------------------------------------------------------
    def _get_observations(self) -> dict:
        # 중간값 계산
        self._compute_intermediate_values()

        # 로봇 팔 joint position
        joint_pos = self._robot.data.joint_pos[:, self.arm_joint_ids]

        # 로봇 팔 joint velocity
        joint_vel = self._robot.data.joint_vel[:, self.arm_joint_ids]

        # finger joint position 2개를 뽑음
        # [a, b] 로 여러 인덱스를 한 번에 선택
        finger_pos = self._robot.data.joint_pos[:, [self.left_finger_id, self.right_finger_id]]

        # finger joint velocity
        finger_vel = self._robot.data.joint_vel[:, [self.left_finger_id, self.right_finger_id]]

        # object의 선속도(linear velocity)
        obj_lin_vel = self._object.data.root_lin_vel_w

        # torch.cat([...], dim=-1)
        # 여러 텐서를 마지막 차원 기준으로 이어붙인다.
        obs = torch.cat(
            [
                joint_pos,       # 7
                joint_vel,       # 7
                finger_pos,      # 2
                finger_vel,      # 2
                self.ee_pos_w,   # 3
                self.obj_pos_w,  # 3
                self.obj_to_ee,  # 3
                obj_lin_vel,     # 3
            ],
            dim=-1,
        )

        # 정책 네트워크가 쓸 observation 반환
        return {"policy": obs}

    # ---------------------------------------------------------------------
    # rewards
    """
    보상 계산
        가까워지면 reach reward
        잡으면 grasp reward
        들면 lift reward
        action 너무 크면 penalty
        joint velocity 너무 크면 penalty
    """
    # Reward = reach + grasp + lift - action_penalty - joint_vel_penalty
    # ---------------------------------------------------------------------
    def _get_rewards(self) -> torch.Tensor:
        # 중간값 최신화
        self._compute_intermediate_values()

        # torch.norm(..., dim=-1)
        # 마지막 차원을 기준으로 벡터의 크기(norm) 계산
        # 즉 ee와 object 사이 거리
        dist = torch.norm(self.obj_to_ee, dim=-1)

        # 1) reach reward
        # 거리가 가까울수록 값이 커짐
        # exp(-6*dist) 형태
        reach_reward = torch.exp(-6.0 * dist)

        # 2) grasp reward
        # 왼손가락 현재 joint position
        left_finger_pos = self._robot.data.joint_pos[:, self.left_finger_id]

        # 오른손가락 현재 joint position
        right_finger_pos = self._robot.data.joint_pos[:, self.right_finger_id]

        # 손가락 평균 위치
        finger_mean = 0.5 * (left_finger_pos + right_finger_pos)

        # 손가락이 충분히 닫혔다고 판단
        gripper_closed = finger_mean < 0.025

        # object에 충분히 가까운지 판단
        near_object = dist < 0.04

        obj_height = self.obj_pos_w[:, 2]

        # 둘 다 만족하면 grasped=True
        self.grasped = gripper_closed & near_object

        bad_close_penalty = (gripper_closed & (~near_object)).float() * 0.2

        # bool -> float 변환
        # True는 1.0, False는 0.0
        grasp_reward = self.cfg.rew_grasp * self.grasped.float()

        # 3) lift reward
        # object의 z 높이
        obj_height = self.obj_pos_w[:, 2]

        # 기본 높이(object_z)보다 얼마나 들렸는지
        # clamp(min=0.0) -> 음수면 0으로 자름
        lifted_height = torch.clamp(obj_height - self.cfg.object_z, min=0.0)

        # torch.where(조건, 참일때값, 거짓일때값)
        # grasped 상태일 때만 lift reward 부여
        lift_reward = torch.where(
            self.grasped,
            self.cfg.rew_lift * lifted_height,
            torch.zeros_like(lifted_height),
        )

        # action penalty 계산
        # self.actions**2 : 각 action 제곱
        # sum(..., dim=-1) : action 차원 기준 합
        action_penalty = (-self.cfg.rew_action_penalty) * torch.sum(self.actions**2, dim=-1)

        # joint velocity penalty 계산
        joint_vel_penalty = (-self.cfg.rew_joint_vel_penalty) * torch.sum(
            self._robot.data.joint_vel[:, self.arm_joint_ids] ** 2,
            dim=-1,
        )

        # 최종 reward 합산
        reward = (
            self.cfg.rew_reach * reach_reward
            + grasp_reward
            + lift_reward
            - action_penalty
            - joint_vel_penalty
            - bad_close_penalty
        )
        return reward

    # ---------------------------------------------------------------------
    # dones
    """
    종료 여부 확인
        너무 오래 했으면 timeout
        충분히 들었으면 success 종료
        물체가 떨어졌으면 실패 종료
    """
    # Grasp RL 초기 단계에서는 너무 공격적인 종료를 피함
    # ---------------------------------------------------------------------
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # 최신 상태 계산
        self._compute_intermediate_values()

        # timeout 조건
        # episode_length_buf 가 최대 길이에 거의 도달했는지 확인
        time_out = self.episode_length_buf >= self.max_episode_length - 1

        # 물체 높이
        obj_height = self.obj_pos_w[:, 2]

        # 성공 조건: 일정 높이 이상 들었으면 성공
        success = obj_height > self.cfg.success_lift_height

        # 실패 조건: 물체가 너무 아래로 떨어졌으면 실패
        object_fallen = obj_height < 0.0

        # 종료 조건
        terminated = success | object_fallen

        # (terminated, time_out) 튜플 반환
        return terminated, time_out

    # ---------------------------------------------------------------------
    # reset
    """
    초기화
        로봇 자세를 기본값 + 약간 랜덤으로 초기화
        물체 위치를 x,y 범위 안에서 랜덤 초기화
    """
    # object는 각 env origin 기준 local randomization
    # ---------------------------------------------------------------------
    def _reset_idx(self, env_ids: Sequence[int] | None):
        # env_ids가 None이면 전체 환경을 리셋
        if env_ids is None:
            env_ids = self._robot._ALL_INDICES

        # 부모 클래스의 reset도 먼저 수행
        super()._reset_idx(env_ids)

        # env_ids를 torch tensor로 변환
        self.env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

        # -----------------------------
        # robot reset
        # -----------------------------

        # 기본 joint position 복사
        default_joint_pos = self._robot.data.default_joint_pos[env_ids].clone()

        # 기본 joint velocity 복사
        default_joint_vel = self._robot.data.default_joint_vel[env_ids].clone()

        # uniform random noise 생성
        # 범위: -0.05 ~ 0.05
        # shape: (리셋할 env 수, 로봇 전체 joint 수)
        noise = sample_uniform(
            -0.05,
            0.05,
            (len(env_ids), self._robot.num_joints),
            self.device,
        )

        # 기본 자세에 노이즈 추가
        joint_pos = default_joint_pos + noise

        # 속도는 기본값 유지
        joint_vel = default_joint_vel

        # joint target 업데이트
        self.joint_targets[env_ids] = joint_pos

        # sim에 joint state 직접 기록
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

        # target도 동일하게 설정
        self._robot.set_joint_position_target(joint_pos, env_ids=env_ids)

        # -----------------------------
        # object reset
        # -----------------------------

        # object 기본 root state 복사
        object_default_state = self._object.data.default_root_state[env_ids].clone()

        # 각 환경의 원점(origin) 좌표
        env_origins = self.scene.env_origins[env_ids]

        # x 랜덤 위치 샘플링
        rand_x = sample_uniform(
            self.cfg.object_x_range[0],
            self.cfg.object_x_range[1],
            (len(env_ids),),
            self.device,
        )

        # y 랜덤 위치 샘플링
        rand_y = sample_uniform(
            self.cfg.object_y_range[0],
            self.cfg.object_y_range[1],
            (len(env_ids),),
            self.device,
        )

        # world position = env origin + local offset
        object_default_state[:, 0] = env_origins[:, 0] + rand_x
        object_default_state[:, 1] = env_origins[:, 1] + rand_y
        object_default_state[:, 2] = env_origins[:, 2] + self.cfg.object_z

        # [7:13] 범위는 linear/angular velocity 영역
        # 모두 0으로 초기화
        object_default_state[:, 7:13] = 0.0

        # 물체 상태를 sim에 기록
        self._object.write_root_state_to_sim(object_default_state, env_ids=env_ids)

        # 중간값 다시 계산
        self._compute_intermediate_values()

    # ---------------------------------------------------------------------
    # intermediate values
    # ---------------------------------------------------------------------
    def _compute_intermediate_values(self):
        # body_pos_w[:, self.ee_body_id, :]
        # 모든 env에 대해 ee body의 world position 추출
        self.ee_pos_w = self._robot.data.body_pos_w[:, self.ee_body_id, :]

        # object world position
        self.obj_pos_w = self._object.data.root_pos_w

        # object -> ee 상대벡터 계산
        self.obj_to_ee = self.obj_pos_w - self.ee_pos_w