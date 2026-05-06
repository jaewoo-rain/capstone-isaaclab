from __future__ import annotations

import torch

from source.omy.vision.tasks.common.omy_base_vision_env import OmyBaseVisionEnv
from source.omy.vision.tasks.common.omy_vision_env_cfg import OmyVisionEnvCfg


class OmyGraspVisionEnv(OmyBaseVisionEnv):
    cfg: OmyVisionEnvCfg

    def _get_rewards(self) -> torch.Tensor:
        # 공통적으로 계산된 상태값들 가져오기
        # ex) 거리, grasp 여부, 정렬 여부, 물체 위치 등
        t = self._get_common_terms()

        # -------------------------
        # 1. grasp 성공 보너스
        # -------------------------
        # 실제로 물체를 잡았으면 큰 보상
        grasp_bonus = t['is_grasping'].float() * 5.0


        # -------------------------
        # 2. 정렬 점수 (핵심)
        # -------------------------
        # xy, z 거리 모두 가까울수록 1에 가까워짐
        # 멀어지면 exponential로 급격히 감소
        alignment_score = torch.exp(-60.0 * t['xy_dist']**2) * torch.exp(-60.0 * t['z_dist']**2)


        # -------------------------
        # 3. 그리퍼 닫기 보상
        # -------------------------
        # 조건:
        # - 정렬이 잘 되어 있어야 함
        # - 그리퍼가 닫히는 방향으로 움직였을 때만 보상
        # - aligned 상태일 때만 활성화
        close_reward = (
            alignment_score
            * torch.clamp(t['gripper_joint'], min=0.0)  # 닫는 방향만 인정
            * t['aligned'].float()
        )


        # -------------------------
        # 4. 사전 준비 보상 (pre-grasp)
        # -------------------------
        # grasp 직전 좋은 위치/자세에 들어오면 보상
        pre_grasp_reward = t['pre_grasp_ready'].float() * 2.0


        # -------------------------
        # 5. 물체 높이 기반 lift 보상
        # -------------------------
        obj_height = t['obj_pos'][:, 2]

        # 바닥 기준이 아니라 "물체 반 높이" 기준으로 lift 계산
        lift_reward = (
            torch.clamp(
                obj_height - (self.cfg.object_size_xyz[2] * 0.5),  # 물체가 실제로 들렸는지
                min=0.0
            )
            * (1.0 + 2.0 * t['is_grasping'].float())  # 잡고 있을 때 더 큰 보상
            * 10.0
        )


        # -------------------------
        # 6. action 패널티
        # -------------------------
        # 로봇 팔(6 DOF)의 과도한 움직임 억제
        action_penalty = torch.sum(self.actions[:, :6] ** 2, dim=-1) * 0.001


        # -------------------------
        # 7. 최종 reward 조합
        # -------------------------
        reward = (
            # 접근 보상
            0.2 * t['approach_reward']

            # xy / z 정렬 보상
            + 0.2 * t['xy_align_reward']
            # + 0.2 * t['z_align_reward']

            # 그리퍼 닫기 (핵심 단계)
            + 2.5 * close_reward

            # grasp 준비 단계
            # + 0.5 * pre_grasp_reward

            # grasp 성공
            + 0.5 * grasp_bonus

            # lift (비중 낮게 시작)
            # + 0.1 * lift_reward

            # -------------------------
            # Vision 관련 보상
            # -------------------------
            # 정상적으로 vision detection 되었을 때 보상
            + 0.25 * t['vision_ok'].float()

            # vision 업데이트 안되면 패널티
            - 0.15 * t['vision_stale'].float()

            # vision miss 누적 패널티 (최대 0.25)
            - torch.clamp(self.vision_miss_count.float() * 0.01, max=0.25)

            # action penalty
            - action_penalty
        )


        # -------------------------
        # 8. 로그 기록 (디버깅용)
        # -------------------------
        self.reward_log = {
            'success_rate': float(t['is_grasping'].float().mean()),  # grasp 성공률
            'vision_ok': float(t['vision_ok'].float().mean()),       # vision 정상 비율
            'vision_stale': float(t['vision_stale'].float().mean())  # vision stale 비율
        }

        # 외부에서 확인 가능하도록 저장
        self.extras['log'] = dict(self.reward_log)

        return reward