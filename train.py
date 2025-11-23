"""Headless training script for StickmanFightEnv using Stable Baselines3 PPO.

Priority optimizations applied:
1. Headless training (no per-step render in training env)
2. Periodic checkpoint saves for external live viewer process
3. Optional parallel environments via SubprocVecEnv for throughput

Control mode note:
    Env now uses motor-based control with 10 target angles (normalized -1..1):
    0 Neck, 1 L Shoulder, 2 R Shoulder, 3 L Elbow, 4 R Elbow,
    5 L Hip, 6 R Hip, 7 L Knee, 8 R Knee, 9 Spine.

Reward rule update:
    - If both agents have equal HP at step end: small negative penalty (-0.01).
    - Otherwise: reward = (hp_self - hp_opp) / MAX_HP (positive if ahead, negative if behind).
    - Tiny standing bonus retained for head height (>120) to discourage staying down.

Run examples (PowerShell):
    python train.py --total-timesteps 600000
    python train.py --num-envs 8 --total-timesteps 1600000 --no-self-play

Launch viewer separately:
    python live_view.py
"""

import os
import argparse
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

from stickfight.env import StickmanFightEnv


def make_env(render: bool = False):
    def _f():
        return StickmanFightEnv(
            render_mode="human" if render else None,
            dummy_policy="random",
            role_swap=True,
            opponent_policy=None,
        )
    return _f


class SelfPlayCallback(BaseCallback):
    def __init__(self, warmup_steps, snapshot_interval, checkpoint_interval, device, verbose=0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.half_warmup_steps = warmup_steps // 2
        self.snapshot_interval = snapshot_interval
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.opponent_model = None
        self.static_attached = False

    def _attach_opponent_policy(self):
        if hasattr(self.training_env, "envs"):
            env0 = self.training_env.envs[0]
            env0.set_opponent_policy(lambda obs: self.opponent_model.predict(obs, deterministic=True))
        elif hasattr(self.training_env, "env_method"):
            # SubprocVecEnv: assign only to first worker
            self.training_env.env_method(
                "set_opponent_policy",
                lambda obs: self.opponent_model.predict(obs, deterministic=True),
                indices=0,
            )
        if self.verbose:
            print("[SelfPlay] Opponent policy attached to env 0")

    def _attach_static_opponent(self):
        if self.static_attached:
            return
        def _static_policy(obs):
            # Return zero action vector (no movement). Mimic SB3 predict signature.
            action_dim = 10  # fixed motor count per env spec
            return np.zeros((action_dim,), dtype=np.float32), None
        if hasattr(self.training_env, "envs"):
            env0 = self.training_env.envs[0]
            env0.set_opponent_policy(_static_policy)
        elif hasattr(self.training_env, "env_method"):
            self.training_env.env_method("set_opponent_policy", _static_policy, indices=0)
        self.static_attached = True
        if self.verbose:
            print(f"[SelfPlay] Static no-movement opponent attached at {self.num_timesteps} timesteps")

    def _make_snapshot(self):
        path = os.path.join("models", "snapshots")
        os.makedirs(path, exist_ok=True)
        snap_path = os.path.join(path, f"opponent_{self.num_timesteps}.zip")
        self.model.save(snap_path)
        self.opponent_model = PPO.load(snap_path, device=self.device)
        self._attach_opponent_policy()
        if self.verbose:
            print(f"[SelfPlay] Snapshot opponent loaded from {snap_path}")

    def _save_checkpoint(self):
        path = os.path.join("models", "checkpoints")
        os.makedirs(path, exist_ok=True)
        ckpt_path = os.path.join(path, f"ckpt_{self.num_timesteps}.zip")
        self.model.save(ckpt_path)
        if self.verbose:
            print(f"[Checkpoint] Saved {ckpt_path}")

    def _on_step(self) -> bool:
        t = self.num_timesteps
        if t == self.half_warmup_steps:
            self._attach_static_opponent()
        if t == self.warmup_steps:
            self._make_snapshot()
        if self.opponent_model and t % self.snapshot_interval == 0 and t > self.warmup_steps:
            self._make_snapshot()
        if t % self.checkpoint_interval == 0:
            self._save_checkpoint()
        return True


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=1, help="Number of parallel envs (SubprocVecEnv if >1)")
    p.add_argument("--total-timesteps", type=int, default=3_000_000, help="Total training timesteps")
    p.add_argument("--n-steps-per-env", type=int, default=256, help="Rollout length per env (adjusted to keep total n_steps divisible)")
    p.add_argument("--no-self-play", action="store_true", help="Disable self-play logic")
    p.add_argument("--warmup-steps", type=int, default=1_000_000)
    p.add_argument("--snapshot-interval", type=int, default=500_000)
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    return p.parse_args()


def build_vec_env(num_envs: int):
    if num_envs == 1:
        return DummyVecEnv([make_env(render=False)])
    return SubprocVecEnv([make_env(render=False) for _ in range(num_envs)])


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}; num_envs={args.num_envs}")

    # Headless training env(s)
    env = build_vec_env(args.num_envs)
    eval_env = DummyVecEnv([make_env(render=False)])

    # Ensure n_steps is divisible by num_envs (SB3 requirement)
    n_steps = args.n_steps_per_env * args.num_envs

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="tb_logs",
        learning_rate=3e-4,  # consider 2e-4 if instability observed with larger action dim
        n_steps=n_steps,
        batch_size=256,
        ent_coef=0.01,
        gamma=0.99,
        device="cpu",
    )

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="models/eval",
        eval_freq=50_000,
        deterministic=True,
        render=False,
    )

    callbacks_list = [eval_cb]
    if not args.no_self_play:
        callbacks_list.append(
            SelfPlayCallback(
                args.warmup_steps,
                args.snapshot_interval,
                args.checkpoint_interval,
                device,
                verbose=1,
            )
        )
    callbacks = CallbackList(callbacks_list)

    model.learn(total_timesteps=args.total_timesteps, callback=callbacks)
    os.makedirs("models/final", exist_ok=True)
    model.save("models/final/ppo_stickfight")
    print("Training complete. Final model saved to models/final/ppo_stickfight.zip")

    # Optional short post-training render demo (single episode) to verify behavior
    try:
        demo_env = StickmanFightEnv(render_mode="human", dummy_policy="random")
        obs, _ = demo_env.reset()
        for _ in range(500):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = demo_env.step(action)
            demo_env.render()
            if term or trunc:
                obs, _ = demo_env.reset()
        demo_env.close()
    except Exception as e:
        print(f"Render demo skipped due to: {e}")


if __name__ == "__main__":
    main()
