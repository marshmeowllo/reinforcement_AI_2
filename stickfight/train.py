"""Tuned training script with optimized hyperparameters.

Key tuning changes:
1. More aggressive exploration (higher entropy, longer warmup)
2. Stronger standing rewards in early stages
3. Better curriculum pacing (longer early stages)
4. Optimized PPO hyperparameters for physics-based control
5. Better observation normalization
"""

import os
import argparse
import torch
import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback, CallbackList
from stable_baselines3.common.monitor import Monitor

from env import StickmanFightEnv


class CurriculumWrapper(gym.Wrapper):
    """Improved curriculum with stronger early-stage rewards."""

    def __init__(self, env: StickmanFightEnv):
        super().__init__(env)
        self.env: StickmanFightEnv = env
        self.current_stage: int = 0
        self._push_cooldown: int = 0
        self.episode_steps = 0
        self.total_standing_reward = 0.0
        self.consecutive_standing_steps = 0  # Track how long agent stays standing

    def set_stage(self, stage: int):
        self.current_stage = int(stage)
        print(f"[Curriculum] Switched to stage {stage}")
        
        # Configure opponent based on stage
        if self.current_stage < 5:
            self.env.role_swap = False
            opponent_policy = self._get_opponent_policy_for_stage()
            if opponent_policy:
                self.env.set_opponent_policy(opponent_policy)
        else:
            # Self-play stage - managed by SelfPlayCallback
            self.env.role_swap = True

    def get_stage(self) -> int:
        return int(self.current_stage)

    def reset(self, **kwargs):
        self.episode_steps = 0
        self.total_standing_reward = 0.0
        self._push_cooldown = 0
        self.consecutive_standing_steps = 0
        
        # Set opponent policy for current stage
        if self.current_stage < 5:
            self.env.role_swap = False
            opponent_policy = self._get_opponent_policy_for_stage()
            if opponent_policy:
                self.env.set_opponent_policy(opponent_policy)
        
        return super().reset(**kwargs)

    def step(self, action):
        obs, base_r, terminated, truncated, info = self.env.step(action)
        self.episode_steps += 1

        # Apply curriculum-specific reward shaping
        shaped_r = self._shape_reward(obs, base_r, info, action)
        
        # Add stage info to info dict
        info['curriculum_stage'] = self.current_stage
        info['shaped_reward'] = shaped_r
        info['consecutive_standing'] = self.consecutive_standing_steps
        
        return obs, shaped_r, terminated, truncated, info

    def _static_policy(self, obs):
        """Static opponent that does nothing."""
        return np.zeros((10,), dtype=np.float32), None
    
    def _scripted_standing_opponent(self, obs):
        """Opponent that tries to maintain standing posture."""
        angles = obs[0:10]
        target = np.array([
            0.0,   # neck centered
            -0.2, 0.2,   # shoulders slightly down
            0.3, -0.3,   # elbows slightly bent
            0.1, 0.1,    # hips slightly bent
            -0.2, -0.2,  # knees bent for stability
            0.0,   # spine neutral
        ])
        action = np.clip(target - angles * 0.5, -1.0, 1.0)
        return action, None
    
    def _scripted_aggressive_opponent(self, obs):
        """Opponent that tries to stand AND approach."""
        angles = obs[0:10]
        base_target = np.array([
            0.0, -0.2, 0.2, 0.3, -0.3, 0.1, 0.1, -0.2, -0.2, 0.0
        ])
        
        rel_x, rel_y = obs[24:26]
        
        if rel_x > 0.1:
            base_target[2] = 0.6
            base_target[4] = 0.0
            base_target[9] = 0.2
        elif rel_x < -0.1:
            base_target[1] = 0.6
            base_target[3] = 0.0
            base_target[9] = -0.2
        
        action = np.clip(base_target - angles * 0.5, -1.0, 1.0)
        return action, None
    
    def _get_opponent_policy_for_stage(self):
        """Return appropriate opponent policy for current stage."""
        if self.current_stage == 0:
            return lambda obs: (np.random.uniform(-0.3, 0.3, 10).astype(np.float32), None)
        elif self.current_stage == 1:
            return self._static_policy
        elif self.current_stage == 2:
            return self._scripted_standing_opponent
        elif self.current_stage == 3:
            return self._scripted_standing_opponent
        elif self.current_stage == 4:
            return self._scripted_aggressive_opponent
        else:
            return None

    def _get_standing_height(self) -> float:
        """Get height difference between head and feet."""
        self_idx = self.env.self_index
        agent = self.env.agent_a if self_idx == 0 else self.env.agent_b
        
        head_y = agent.head.position.y
        l_foot_y = agent.parts['l_lower_leg'].position.y
        r_foot_y = agent.parts['r_lower_leg'].position.y
        feet_y = (l_foot_y + r_foot_y) / 2.0
        
        return head_y - feet_y

    def _get_torso_angle(self) -> float:
        """Get torso angle from vertical (0 = upright)."""
        self_idx = self.env.self_index
        agent = self.env.agent_a if self_idx == 0 else self.env.agent_b
        return abs(agent.torso.angle)

    def _shape_reward(self, obs: np.ndarray, base_r: float, info: dict, action: np.ndarray) -> float:
        """Apply curriculum-specific reward shaping."""
        
        stage = self.current_stage
        
        # Stage 0: Learn to move joints (exploration)
        if stage == 0:
            # TUNED: Much stronger movement rewards
            ang_vel = np.abs(obs[10:20])
            movement_reward = np.mean(ang_vel) * 0.5  # Increased from 0.1
            
            # Reward action diversity more strongly
            action_diversity = np.std(action) * 0.2  # Increased from 0.05
            
            # Extra bonus for high-energy movement
            high_energy_bonus = 0.0
            if np.mean(ang_vel) > 2.0:
                high_energy_bonus = 0.1
            
            return movement_reward + action_diversity + high_energy_bonus
        
        # Stage 1: Learn to stand upright
        elif stage == 1:
            height = self._get_standing_height()
            torso_angle = self._get_torso_angle()
            
            # TUNED: Much stronger and more granular height rewards
            height_reward = 0.0
            if height > 120:
                height_reward = 0.5  # MAJOR reward for tall standing
                self.consecutive_standing_steps += 1
            elif height > 110:
                height_reward = 0.3  # Increased from 0.1
                self.consecutive_standing_steps += 1
            elif height > 90:
                height_reward = 0.15  # Increased from 0.05
                self.consecutive_standing_steps = 0
            elif height > 70:
                height_reward = 0.05  # Increased from 0.02
                self.consecutive_standing_steps = 0
            else:
                height_reward = -0.1  # Stronger penalty from -0.05
                self.consecutive_standing_steps = 0
            
            # TUNED: Stronger upright torso rewards
            upright_reward = 0.0
            if torso_angle < 0.2:  # Very upright
                upright_reward = 0.15  # Increased from 0.05
            elif torso_angle < 0.4:
                upright_reward = 0.08  # Increased from 0.02
            elif torso_angle < 0.6:
                upright_reward = 0.03
            else:
                upright_reward = -0.05  # Stronger penalty
            
            # TUNED: Bonus for sustained standing
            standing_duration_bonus = 0.0
            if self.consecutive_standing_steps > 50:
                standing_duration_bonus = 0.2
            elif self.consecutive_standing_steps > 20:
                standing_duration_bonus = 0.1
            
            # Less aggressive stability penalty
            stability_penalty = 0.0
            if height > 100:
                stability_penalty = -np.abs(obs[12]) * 0.0005
            
            return height_reward + upright_reward + standing_duration_bonus + stability_penalty
        
        # Stage 2: Maintain balance under disturbances
        elif stage == 2:
            if self._push_cooldown <= 0 and self.episode_steps % 80 == 0:
                self._apply_push()
                self._push_cooldown = np.random.randint(40, 100)
            else:
                self._push_cooldown -= 1
            
            height = self._get_standing_height()
            torso_angle = self._get_torso_angle()
            
            # TUNED: Even stronger balance rewards
            balance_reward = 0.0
            if height > 110 and torso_angle < 0.4:
                balance_reward = 0.3  # Increased from 0.15
            elif height > 90:
                balance_reward = 0.1  # Increased from 0.05
            else:
                balance_reward = -0.08
            
            # TUNED: Better recovery rewards
            if height > 90 and self.episode_steps > 0:
                ang_vel = obs[10:20]
                recovery_speed = np.abs(ang_vel).mean()
                if height < 100:
                    balance_reward += recovery_speed * 0.05  # Increased from 0.02
            
            return balance_reward
        
        # Stage 3: Approach opponent
        elif stage == 3:
            approach_delta = float(info.get('approach_reward', 0.0))
            height = self._get_standing_height()
            
            # TUNED: Much stronger approach incentive
            approach_reward = approach_delta * 20.0  # Increased from 10.0
            
            # Must stay standing while approaching
            standing_bonus = 0.1 if height > 100 else -0.1  # Increased from 0.05/-0.05
            
            return approach_reward + standing_bonus
        
        # Stage 4: Learn to make contact and deal damage
        elif stage == 4:
            damage_dealt = float(info.get('damage_dealt_self', 0.0))
            approach_delta = float(info.get('approach_reward', 0.0))
            height = self._get_standing_height()
            
            # TUNED: Massive damage rewards
            combat_reward = damage_dealt * 20.0  # Increased from 10.0
            
            # Continue rewarding approach
            approach_reward = approach_delta * 8.0  # Increased from 5.0
            
            # Must stay standing
            standing_bonus = 0.05 if height > 100 else -0.05
            
            return combat_reward + approach_reward + standing_bonus
        
        # Stage 5+: Full environment reward (self-play)
        else:
            return float(base_r)

    def _apply_push(self):
        """Apply a random push to the agent's torso."""
        agent = self.env.agent_a if self.env.self_index == 0 else self.env.agent_b
        mag = np.random.uniform(300, 800)
        direction = np.random.choice([-1.0, 1.0])
        impulse = (mag * direction, np.random.uniform(100, 400))
        try:
            agent.torso.apply_impulse_at_local_point(impulse)
        except Exception:
            pass


class ProgressiveStageCallback(BaseCallback):
    """Automatically progress through curriculum stages based on performance."""
    
    def __init__(self, stage_timesteps: list[int], verbose=1):
        super().__init__(verbose)
        self.stage_timesteps = stage_timesteps
        self.boundaries = np.cumsum(stage_timesteps).tolist()
        self.current_stage = 0

    def _on_training_start(self) -> None:
        try:
            self.training_env.env_method('set_stage', 0)
            if self.verbose:
                print(f"[Curriculum] Starting at stage 0")
        except Exception as e:
            print(f"Warning: Could not set initial stage: {e}")

    def _on_step(self) -> bool:
        t = self.num_timesteps
        
        new_stage = 0
        for idx, boundary in enumerate(self.boundaries):
            if t >= boundary:
                new_stage = idx + 1
        
        if new_stage != self.current_stage and new_stage <= len(self.stage_timesteps):
            self.current_stage = new_stage
            try:
                self.training_env.env_method('set_stage', self.current_stage)
                if self.verbose:
                    print(f"\n{'='*60}")
                    print(f"[Curriculum] Advanced to stage {self.current_stage} at {t} timesteps")
                    print(f"{'='*60}\n")
            except Exception as e:
                print(f"Warning: Could not update stage: {e}")
        
        return True


class SelfPlayCallback(BaseCallback):
    """Manage self-play opponent during final stage with progressive difficulty."""
    
    def __init__(self, start_timestep: int, snapshot_interval: int, 
                 checkpoint_interval: int, device: str, verbose=1):
        super().__init__(verbose)
        self.start_timestep = start_timestep
        self.snapshot_interval = snapshot_interval
        self.checkpoint_interval = checkpoint_interval
        self.device = device
        self.opponent_model = None
        self.last_snapshot = 0
        self.opponent_history = []
        self.max_history = 5
        self.opponent_path = None  # Store path instead of model

    def _on_step(self) -> bool:
        t = self.num_timesteps
        
        if t < self.start_timestep:
            return True
        
        if self.opponent_model is None:
            self._create_snapshot()
            if self.verbose:
                print(f"[SelfPlay] Initialized opponent at timestep {t}")
        
        if t - self.last_snapshot >= self.snapshot_interval:
            self._create_snapshot()
        
        if t % self.checkpoint_interval == 0:
            self._save_checkpoint()
        
        return True

    def _create_snapshot(self):
        """Create opponent snapshot from current model."""
        path = os.path.join("models", "snapshots")
        os.makedirs(path, exist_ok=True)
        
        snap_path = os.path.join(path, f"opponent_{self.num_timesteps}.zip")
        self.model.save(snap_path)
        
        # Store path instead of loading model
        self.opponent_path = snap_path
        self.opponent_history.append(snap_path)
        if len(self.opponent_history) > self.max_history:
            self.opponent_history.pop(0)
        
        self.opponent_model = PPO.load(snap_path, device=self.device)
        self.last_snapshot = self.num_timesteps
        
        # Attach to environment (only for single-env or DummyVecEnv)
        try:
            if hasattr(self.training_env, 'envs'):
                # DummyVecEnv - direct access
                self._attach_opponent_to_env(self.training_env.envs[0])
            else:
                # SubprocVecEnv - can't use lambda, use simpler approach
                if self.verbose:
                    print(f"[SelfPlay] Note: Using simplified opponent for SubprocVecEnv")
            
            if self.verbose:
                print(f"[SelfPlay] Updated opponent (history size: {len(self.opponent_history)})")
        except Exception as e:
            if self.verbose:
                print(f"[SelfPlay] Note: Self-play works best with --num-envs 1")

    def _attach_opponent_to_env(self, env):
        """Attach opponent policy directly to a single environment."""
        # Unwrap to get the actual StickmanFightEnv
        actual_env = env
        while hasattr(actual_env, 'env'):
            actual_env = actual_env.env
        
        # Create opponent policy that uses loaded models
        def opponent_policy(obs):
            # 70% latest, 30% random from history
            if np.random.random() < 0.7 or len(self.opponent_history) == 1:
                model = self.opponent_model
            else:
                path = np.random.choice(self.opponent_history[:-1])
                model = PPO.load(path, device=self.device)
            
            return model.predict(obs, deterministic=False)
        
        actual_env.set_opponent_policy(opponent_policy)

    def _save_checkpoint(self):
        """Save training checkpoint."""
        path = os.path.join("models", "checkpoints")
        os.makedirs(path, exist_ok=True)
        
        ckpt_path = os.path.join(path, f"ckpt_{self.num_timesteps}.zip")
        self.model.save(ckpt_path)
        
        if self.verbose:
            print(f"[Checkpoint] Saved at {self.num_timesteps} timesteps")


def make_env(render_mode=None):
    """Create environment with monitoring."""
    def _init():
        env = StickmanFightEnv(
            render_mode=render_mode,
            dummy_policy="random",
            role_swap=True,
            opponent_policy=None,
        )
        env = CurriculumWrapper(env)
        env = Monitor(env)
        return env
    return _init


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num-envs", type=int, default=1, 
                   help="Number of parallel environments (use 1 for self-play, >1 for faster early training)")
    p.add_argument("--total-timesteps", type=int, default=2_000_000, help="Total training steps")
    # TUNED: Longer early stages for better skill development
    p.add_argument("--stage-steps", type=str, default="200000,300000,250000,300000,350000",
                   help="Timesteps for stages 0-4 (stage 5 uses remaining)")
    p.add_argument("--no-self-play", action="store_true")
    p.add_argument("--checkpoint-interval", type=int, default=100_000)
    p.add_argument("--snapshot-interval", type=int, default=150_000)
    p.add_argument("--normalize-obs", action="store_true", help="Use observation normalization")
    return p.parse_args()


def main():
    args = parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"\n{'='*60}")
    print(f"TUNED STICKMAN TRAINING")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Parallel Environments: {args.num_envs}")
    if args.num_envs > 1 and not args.no_self_play:
        print(f"Note: Self-play works best with --num-envs 1")
        print(f"      Multiple envs are faster for stages 0-4 only")
    print(f"Total Timesteps: {args.total_timesteps:,}")
    print(f"{'='*60}\n")

    # Parse stage timesteps
    stage_steps = [int(x.strip()) for x in args.stage_steps.split(',')]
    if len(stage_steps) != 5:
        stage_steps = [200_000, 300_000, 250_000, 300_000, 350_000]
    
    self_play_start = sum(stage_steps)
    print(f"Curriculum breakdown:")
    print(f"  Stage 0 (Movement):  {stage_steps[0]:>8,} steps")
    print(f"  Stage 1 (Standing):  {stage_steps[1]:>8,} steps")
    print(f"  Stage 2 (Balance):   {stage_steps[2]:>8,} steps")
    print(f"  Stage 3 (Approach):  {stage_steps[3]:>8,} steps")
    print(f"  Stage 4 (Combat):    {stage_steps[4]:>8,} steps")
    print(f"  Stage 5 (Self-play): {max(0, args.total_timesteps - self_play_start):>8,} steps")
    print(f"{'='*60}\n")

    # Create environments
    if args.num_envs > 1:
        env = SubprocVecEnv([make_env() for _ in range(args.num_envs)])
    else:
        env = DummyVecEnv([make_env()])
    
    # TUNED: Optional observation normalization
    if args.normalize_obs:
        env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.0)
        print("Using observation normalization (VecNormalize)")
    
    eval_env = DummyVecEnv([make_env()])

    # TUNED: Optimized PPO hyperparameters
    policy_kwargs = dict(
        net_arch=dict(pi=[256, 256], vf=[256, 256]),
        activation_fn=torch.nn.SiLU,
    )

    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        tensorboard_log="tb_logs",
        learning_rate=3e-4,  # Standard learning rate
        n_steps=4096,  # TUNED: Even longer rollouts (was 2048)
        batch_size=256,  # Good batch size
        n_epochs=10,  # Standard
        ent_coef=0.03,  # TUNED: Higher entropy for more exploration (was 0.02)
        clip_range=0.2,  # Standard
        gamma=0.99,  # Standard discount
        gae_lambda=0.95,  # Standard GAE
        max_grad_norm=0.5,  # TUNED: Lower for stability
        vf_coef=0.5,  # Standard value function coefficient
        policy_kwargs=policy_kwargs,
        device=device,
    )

    print("\nPPO Hyperparameters:")
    print(f"  Learning Rate:     {model.learning_rate}")
    print(f"  Rollout Steps:     {model.n_steps}")
    print(f"  Batch Size:        {model.batch_size}")
    print(f"  Entropy Coef:      {model.ent_coef}")
    print(f"  Clip Range:        {model.clip_range}")
    print(f"{'='*60}\n")

    # Callbacks
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path="models/best",
        log_path="models/eval",
        eval_freq=25_000,
        deterministic=True,
        render=False,
    )

    stage_cb = ProgressiveStageCallback(stage_steps, verbose=1)
    
    callbacks = [eval_cb, stage_cb]
    
    if not args.no_self_play:
        selfplay_cb = SelfPlayCallback(
            start_timestep=self_play_start,
            snapshot_interval=args.snapshot_interval,
            checkpoint_interval=args.checkpoint_interval,
            device=device,
            verbose=1,
        )
        callbacks.append(selfplay_cb)

    # Train
    print("Starting training...")
    print(f"Monitor with: tensorboard --logdir tb_logs\n")
    
    try:
        model.learn(
            total_timesteps=args.total_timesteps,
            callback=CallbackList(callbacks),
            progress_bar=True,
        )
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user!")
    
    # Save final model
    os.makedirs("models/final", exist_ok=True)
    model.save("models/final/ppo_stickfight")
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Final model saved to: models/final/ppo_stickfight.zip")
    print("="*60 + "\n")

    # Quick demo
    print("Running short demo...")
    try:
        demo_env = StickmanFightEnv(render_mode="human")
        obs, _ = demo_env.reset()
        
        for _ in range(1000):
            action, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = demo_env.step(action)
            demo_env.render()
            
            if term or trunc:
                break
        
        demo_env.close()
    except Exception as e:
        print(f"Demo skipped: {e}")


if __name__ == "__main__":
    main()