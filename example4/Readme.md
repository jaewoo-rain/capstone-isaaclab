1.  reward = (
        + 0.15 * xy_align_reward
        + 0.12 * z_align_reward
        + 10 * close_reward
        + 30 * lift_reward
        + 1000 * success_reward
        - 0.001 * action_penalty
    )
    5,000,000 steps
2.  reward = (
        + 10 * close_reward
        + 30 * lift_reward
        + 1000 * success_reward
        - 0.001 * action_penalty
    )
    10,000,000 steps

-> 집기 성공