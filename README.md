# StickFight Multi-Agent RL Environment

Physics-based 1v1 stickman fighting game environment built with Pymunk (physics) + Pygame (rendering) + Gymnasium (RL interface) + Stable Baselines3 (PPO training).

## Features
- Ragdoll stickmen (head, torso, arms, legs) with pivot + rotary limit joints.
- Motorized joints (9 controllable DOFs per agent): Neck, Left/Right Shoulders, Left/Right Elbows, Left/Right Hips, Left/Right Knees.
- Continuous action space in [-1,1] mapped to motor rates.
- Collision-based damage (limbs vs vulnerable parts) scaled by relative impact velocity.
- Knockback impulses applied on successful hits.
- Reward shaping: damage dealt (+), damage taken (-), standing encouragement (+ small).
- Optional role swapping: each episode randomly picks perspective agent, enabling early self-play style training with a single shared policy.
- Self-play warmup path: train vs random for WARMUP_STEPS, then snapshot policy periodically as opponent (SNAPSHOT_INTERVAL) while saving checkpoints (CHECKPOINT_INTERVAL).
- Normalized observations: joint angles/velocities, torso core state, relative opponent key points, HP values.
- Physics stepping decoupled from render FPS (multiple sub-steps per env step).

## Observation Vector Layout
1. Joint angles (9) normalized by π.
2. Joint angular velocity differences (9) scaled by 1/10.
3. Core: torso angle, torso vx, torso vy, feet-on-ground flag.
4. Relative opponent head (x,y) and torso (x,y) normalized to [-1,1].
5. Relative opponent limbs: left/right hands & feet (8 values).
6. HP (self, opponent) normalized [0,1].

Total length: 9 + 9 + 4 + 4 + 8 + 2 = 36.

## Quick Start
```powershell
# (Optional) Activate virtual environment
# .\myenv\Scripts\Activate.ps1

# Install deps (ensure corrected package name for SB3)
pip install -r requirements.txt

# Train PPO (warmup + self-play + periodic checkpoints)
python -m train
```

## Faster Training & Live Visualization
Headless training removes per-step rendering overhead. A separate viewer process periodically loads checkpoints for real-time visualization.

```powershell
# Headless single-env training (self-play enabled)
python train.py --total-timesteps 600000

# Parallel headless training (8 envs, disable self-play to simplify)
python train.py --num-envs 8 --no-self-play --total-timesteps 1600000

# Launch live viewer in separate terminal (polls checkpoints)
python live_view.py --interval 5 --episode-steps 400
```

Key points:
- Checkpoints saved in `models/checkpoints/` every `checkpoint_interval` (default 100k steps).
- Snapshots for opponent self-play stored in `models/snapshots/` after warmup.
- Viewer only reloads when a newer `.zip` appears; adjust polling `--interval`.
- `--n-steps-per-env` controls rollout length per environment (total `n_steps = num_envs * n_steps_per_env`).
- Use `--no-self-play` when scaling to many envs if opponent snapshot logic is unnecessary.

Optional: decrease `PHYSICS_STEPS_PER_ENV_STEP` or increase `TIME_STEP` in `stickfight/env.py` for faster (but less precise) simulation—retrain from scratch after physics changes.

## Rendering Demo Only
```powershell
python - <<'PY'
from stickfight.env import StickmanFightEnv
import numpy as np
env = StickmanFightEnv(render_mode='human', dummy_policy='random')
obs,_ = env.reset()
for _ in range(500):
    action = np.zeros(9, dtype=np.float32)  # idle
    obs, r, term, trunc, info = env.step(action)
    env.render()
    if term or trunc:
        obs,_ = env.reset()
env.close()
PY
```

## Extending
- Replace `dummy_policy` with a learned second agent (wrap in multi-agent training or self-play loop).
- Set `role_swap=True` (default) to train a perspective-invariant policy; disable by passing `role_swap=False` to focus on one spawn side.
- Self-play tuning: edit `train.py` constants `WARMUP_STEPS`, `SNAPSHOT_INTERVAL`, `CHECKPOINT_INTERVAL`. Snapshots saved under `models/snapshots/`, checkpoints under `models/checkpoints/`, best eval under `models/best/`, final model under `models/final/`.
- Adjust `DAMAGE_CONSTANT`, `KNOCKBACK_SCALE`, joint limits, or motor rates for different fight dynamics.
- Add additional sensors (e.g., distance to ground, limb velocities) by extending `_get_observation`.

## Notes
- Coordinates: Pymunk y increases upward; rendering flips y for Pygame.
- Standing reward uses head height heuristic; tune threshold if needed.
- Stability: `PHYSICS_STEPS_PER_ENV_STEP * TIME_STEP` defines real simulated time per RL step.
- For curriculum, start with static opponent (`dummy_policy='static'`) then move to random.

## License
No explicit license added; adapt as needed.
