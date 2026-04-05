import argparse
import os
import sys
import time
from datetime import datetime

from isaaclab.app import AppLauncher

# ----------------------------
# app launch first
# ----------------------------
parser = argparse.ArgumentParser(description="Train or resume PPO for Example1 tasks")
parser.add_argument("--task", type=str, choices=["reach", "grasp", "lift"], default="reach")
parser.add_argument("--num_envs", type=int, default=None)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ----------------------------
# imports after launch
# ----------------------------
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from isaaclab_rl.sb3 import Sb3VecEnvWrapper


class ProgressFileLogger:
    def __init__(self, log_path: str):
        self.log_path = log_path
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        with open(self.log_path, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("Example1 Training Log\n")
            f.write(f"Created at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

    def write_line(self, text: str):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(text + "\n")

    def write_block(self, lines: list[str]):
        with open(self.log_path, "a", encoding="utf-8") as f:
            for line in lines:
                f.write(line + "\n")
            f.write("\n")


class ProgressCallback(BaseCallback):
    def __init__(
        self,
        total_timesteps: int,
        logger: ProgressFileLogger,
        print_freq: int = 10000,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.total_timesteps_target = total_timesteps
        self.print_freq = print_freq
        self.start_time = None
        self.logger_file = logger

    def _on_training_start(self) -> None:
        self.start_time = time.time()
        self.logger_file.write_block([
            "[INFO] Training started",
            f"task total timesteps: {self.total_timesteps_target}",
            f"print frequency: every {self.print_freq} steps",
        ])

    def _format_eta(self, eta_sec: float) -> str:
        if eta_sec == float("inf"):
            return "inf"
        h = int(eta_sec // 3600)
        m = int((eta_sec % 3600) // 60)
        s = int(eta_sec % 60)
        return f"{h:02d}:{m:02d}:{s:02d}"

    def _on_step(self) -> bool:
        current_step = self.num_timesteps
        remaining = max(self.total_timesteps_target - current_step, 0)
        percent = (current_step / self.total_timesteps_target) * 100 if self.total_timesteps_target > 0 else 0.0

        if current_step > 0 and current_step % self.print_freq == 0:
            elapsed = time.time() - self.start_time if self.start_time is not None else 0.0
            steps_per_sec = current_step / elapsed if elapsed > 0 else 0.0
            eta_sec = remaining / steps_per_sec if steps_per_sec > 0 else float("inf")
            eta_str = self._format_eta(eta_sec)

            lines = [
                "-" * 80,
                "[TRAIN PROGRESS]",
                f"step            : {current_step}/{self.total_timesteps_target}",
                f"progress        : {percent:.2f}%",
                f"remaining_steps : {remaining}",
                f"speed           : {steps_per_sec:.1f} steps/s",
                f"ETA             : {eta_str}",
                f"time            : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            ]

            print("\n".join(lines))
            self.logger_file.write_block(lines)

        return True

    def _on_training_end(self) -> None:
        self.logger_file.write_block([
            "[INFO] Training finished",
            f"final timestep reached: {self.num_timesteps}",
            f"finished at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ])


def build_env_and_cfg(task_name: str):
    if task_name == "reach":
        import source.example1.tasks.reach
        from source.example1.tasks.reach.reach_env_cfg import ReachEnvCfg
        return "Example1-Reach-Franka-v0", ReachEnvCfg()
    elif task_name == "grasp":
        import source.example1.tasks.grasp
        from source.example1.tasks.grasp.grasp_env_cfg import GraspEnvCfg
        return "Example1-Grasp-Franka-v0", GraspEnvCfg()
    else:
        import source.example1.tasks.lift
        from source.example1.tasks.lift.lift_env_cfg import LiftEnvCfg
        return "Example1-Lift-Franka-v0", LiftEnvCfg()


def main():
    env_id, cfg = build_env_and_cfg(args_cli.task)

    if args_cli.num_envs is not None:
        cfg.scene.num_envs = args_cli.num_envs

    env = gym.make(env_id, cfg=cfg)
    env = Sb3VecEnvWrapper(env)

    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    os.makedirs("train_logs", exist_ok=True)

    n_steps = 64
    batch_size = 4096
    learning_rate = 3e-4

    total_timesteps = args_cli.max_iterations * n_steps * cfg.scene.num_envs

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    txt_log_path = f"train_logs/{args_cli.task}_train_log_{timestamp}.txt"
    txt_logger = ProgressFileLogger(txt_log_path)

    txt_logger.write_block([
        "[RUN CONFIG]",
        f"task            : {args_cli.task}",
        f"num_envs        : {cfg.scene.num_envs}",
        f"max_iterations  : {args_cli.max_iterations}",
        f"total_timesteps : {total_timesteps}",
        f"resume          : {args_cli.resume}",
        f"checkpoint      : {args_cli.checkpoint}",
        f"log_file        : {txt_log_path}",
    ])

    progress_callback = ProgressCallback(
        total_timesteps=total_timesteps,
        logger=txt_logger,
        print_freq=10000,
    )

    callback = CallbackList([
        progress_callback,
    ])

    default_checkpoint = f"checkpoints/example1_{args_cli.task}_ppo.zip"
    checkpoint_path = args_cli.checkpoint if args_cli.checkpoint is not None else default_checkpoint

    if args_cli.resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"resume checkpoint not found: {checkpoint_path}")

        print(f"[INFO] Resume training from: {checkpoint_path}")
        txt_logger.write_line(f"[INFO] Resume training from: {checkpoint_path}")

        model = PPO.load(
            checkpoint_path,
            env=env,
            device="cuda",
        )
    else:
        print(f"[INFO] Start new training: task={args_cli.task}, num_envs={cfg.scene.num_envs}")
        txt_logger.write_line(f"[INFO] Start new training: task={args_cli.task}, num_envs={cfg.scene.num_envs}")

        model = PPO(
            "MlpPolicy",
            env,
            n_steps=n_steps,
            batch_size=batch_size,
            learning_rate=learning_rate,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            verbose=1,
            device="cuda",
            tensorboard_log=f"./logs/example1_{args_cli.task}_ppo",
        )

    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        reset_num_timesteps=not args_cli.resume,
    )

    final_path = f"checkpoints/example1_{args_cli.task}_ppo"
    model.save(final_path)
    print(f"[INFO] Final model saved to: {final_path}.zip")
    txt_logger.write_line(f"[INFO] Final model saved to: {final_path}.zip")
    txt_logger.write_line(f"[INFO] Human-readable training log saved to: {txt_log_path}")

    env.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()