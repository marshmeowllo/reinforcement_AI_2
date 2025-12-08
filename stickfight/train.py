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
import gymnasium as gym

from stickfight.env import StickmanFightEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList

# -------------------------
# Curriculum: Wrappers & CB
# -------------------------

class CurriculumWrapper(gym.Wrapper):
    """Stage-based curriculum wrapper.

    Stages:
      0: Exploration (move joints)
      1: Standing (upright posture)
      2: Balance (random pushes)
      3: Approach (move toward opponent)
      4: Contact/Damage (reward damage)
      5: Self-Play (use env reward)
    """

    def __init__(self, env: StickmanFightEnv):
        super().__init__(env)
        self.current_stage = 0
        self._push_cooldown = 0

    # Methods callable via VecEnv.env_method
    def set_stage(self, stage: int):
        self.current_stage = int(stage)
        # opponent policy & role swap policy per stage
        if self.current_stage < 5:
            # Static opponent for early stages
            self.env.role_swap = False
            self.env.set_opponent_policy(self._static_policy)
        else:
            # Hand control back to self-play callback
            self.env.role_swap = True
            self.env.set_opponent_policy(None)

    def get_stage(self) -> int:
        return int(self.current_stage)

    def reset(self, **kwargs):
        # Ensure correct opponent/roles before episode
        if self.current_stage < 5:
            self.env.role_swap = False
            self.env.set_opponent_policy(self._static_policy)
        else:
            self.env.role_swap = True
        self._push_cooldown = 0
        return super().reset(**kwargs)

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)

        # Optionally apply external disturbances in Stage 2
        if self.current_stage == 2 and not (terminated or truncated):
            self._maybe_apply_push()

        shaped_r = self._shape_reward(self.current_stage, obs, base_r, info, action)
        return obs, shaped_r, terminated, truncated, info

    # ---- helpers ----
    def _static_policy(self, obs):
        # SB3 predict-like signature: returns (action, state)
        return np.zeros((10,), dtype=np.float32), None

    def _standing_metric(self) -> float:
        # Head height relative to feet as in env reward
        self_idx = self.env.self_index
        agent = self.env.agent_a if self_idx == 0 else self.env.agent_b
        head_y = agent.head.position.y
        l_foot_y = agent.parts['l_lower_leg'].position.y
        r_foot_y = agent.parts['r_lower_leg'].position.y
        feet_y = (l_foot_y + r_foot_y) / 2.0
        return head_y - feet_y

    def _approach_delta(self, info: dict) -> float:
        # Env already computes approach_reward each step
        return float(info.get('approach_reward', 0.0))

    def _damage_dealt(self, info: dict) -> float:
        return float(info.get('damage_dealt_self', 0.0))

    def _maybe_apply_push(self):
        # Apply a strong but sparse random impulse to the perspective agent torso
        if self._push_cooldown > 0:
            self._push_cooldown -= 1
            return
        # Roughly one push every ~1-2 seconds of env time
        self._push_cooldown = np.random.randint(60, 160)
        agent = self.env.agent_a if self.env.self_index == 0 else self.env.agent_b
        mag = np.random.uniform(500, 1200)
        direction = np.random.choice([-1.0, 1.0])
        impulse = (mag * direction, np.random.uniform(200, 600))
        try:
            agent.torso.apply_impulse_at_local_point(impulse)
        except Exception:
            pass

    def _shape_reward(self, stage: int, obs: np.ndarray, base_r: float, info: dict, action: np.ndarray) -> float:
        if stage == 0:
            # Encourage joint movement (angular velocities in obs indices 10..19)
            ang_vel = np.abs(obs[10:20]).mean()
            act_mag = np.abs(action).mean()
            return 0.5 * ang_vel + 0.5 * act_mag * 0.5
        if stage == 1:
            # Standing upright
            h = self._standing_metric()
            return 0.02 * h - 0.5  # centered around ~0 when low
        if stage == 2:
            # Standing + recovering from pushes
            h = self._standing_metric()
            vel_pen = np.abs(obs[12]) * 0.01  # |vy| small penalty
            return 0.02 * h - 0.5 - vel_pen
        if stage == 3:
            # Move toward opponent (use env approach delta aggressively)
            return 5.0 * self._approach_delta(info)
        if stage == 4:
            # Reward damage + some approach
            return 2.0 * self._damage_dealt(info) + 2.0 * self._approach_delta(info)
        # stage >= 5 -> use environment's combat reward
        return float(base_r)


class CurriculumStageCallback(BaseCallback):
    def __init__(self, stage_steps: list[int], verbose=0):
        super().__init__(verbose)
        self.stage_steps = stage_steps
        self.boundaries = np.cumsum(stage_steps).tolist()
        self.current_stage = 0

    def _stage_from_t(self, t: int) -> int:
        for idx, b in enumerate(self.boundaries):
            if t < b:
                return idx
        return len(self.stage_steps)  # final stage index (5)

    def _on_training_start(self) -> None:
        # Initialize all workers to stage 0
        try:
            self.training_env.env_method('set_stage', 0)
        except Exception:
            pass

    def _on_step(self) -> bool:
        t = self.num_timesteps
        stage = self._stage_from_t(t)
        if stage != self.current_stage:
            self.current_stage = stage
            try:
                self.training_env.env_method('set_stage', int(stage))
            except Exception:
                pass
            if self.verbose:
                print(f"[Curriculum] Switched to stage {stage} at {t} timesteps")
        return True


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
    def __init__(self, warmup_steps, snapshot_interval, checkpoint_interval, device, start_timestep: int = 0, verbose=0):
        super().__init__(verbose)
        self.warmup_steps = warmup_steps
        self.half_warmup_steps = warmup_steps // 2
        self.snapshot_interval = snapshot_interval
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.opponent_model = None
        self.static_attached = False
        self.start_timestep = int(start_timestep)

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
        # Do nothing until curriculum enters self-play stage
        if t < self.start_timestep:
            return True
        # Initialize self-play opponent as soon as we enter self-play
        if self.opponent_model is None:
            self._make_snapshot()
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
    p.add_argument(
        "--stage-steps",
        type=str,
        default="200000,200000,200000,300000,300000",
        help="Comma-separated timesteps for stages 0..4; stage 5 uses the remaining",
    )
    return p.parse_args()


def build_vec_env(num_envs: int):
    def _make_wrapped():
        return CurriculumWrapper(make_env(render=False)())
    if num_envs == 1:
        return DummyVecEnv([_make_wrapped])
    return SubprocVecEnv([_make_wrapped for _ in range(num_envs)])


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}; num_envs={args.num_envs}")

    # Headless training env(s)
    env = build_vec_env(args.num_envs)
    eval_env = DummyVecEnv([make_env(render=False)])

    # Ensure n_steps is divisible by num_envs (SB3 requirement)
    n_steps = args.n_steps_per_env * args.num_envs

    # Parse curriculum stage steps (0..4)
    try:
        stage_steps = [int(x.strip()) for x in args.stage_steps.split(',') if x.strip()]
    except Exception:
        stage_steps = [200_000, 200_000, 200_000, 300_000, 300_000]
    if len(stage_steps) != 5:
        stage_steps = (stage_steps + [200_000]*5)[:5]
    self_play_start_t = sum(stage_steps)

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

    # Curriculum stage scheduler
    curr_cb = CurriculumStageCallback(stage_steps=stage_steps, verbose=1)

    callbacks_list = [eval_cb, curr_cb]
    if not args.no_self_play:
        callbacks_list.append(
            SelfPlayCallback(
                args.warmup_steps,
                args.snapshot_interval,
                args.checkpoint_interval,
                device,
                start_timestep=self_play_start_t,
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
