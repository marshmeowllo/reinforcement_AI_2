"""Run a quick interactive StickmanFightEnv session.

Environment now uses motor-based control:
    Action space = 10 target angles for joints (normalized -1..1).

Reward rule:
    - Tie HP: -0.01 penalty.
    - Advantage: (hp_self - hp_opp) / MAX_HP + tiny standing bonus.

Usage:
    powershell> python run_game.py
"""
from env import StickmanFightEnv
import numpy as np
import random

def randomize_base_torques(env):
    # Randomize persistent base torques used each step by the PD controller
    for agent in (env.agent_a, env.agent_b):
        for i in range(len(agent.base_motor_forces)):
            agent.base_motor_forces[i] *= random.uniform(0.7, 1.3)

def main():
    # Use random dummy so the opponent keeps moving too
    env = StickmanFightEnv(render_mode="human", dummy_policy="random", role_swap=False)
    obs, _ = env.reset()
    # Randomize base torques (persist across steps)
    randomize_base_torques(env)
    # Use smooth random actions so the agent moves
    act_dim = env.action_space.shape[0]
    current_action = np.zeros(act_dim, dtype=np.float32)
    noise_scale = 0.15
    for step in range(600):
        # Random-walk the action for continuous movement
        current_action = np.clip(
            current_action + np.random.uniform(-noise_scale, noise_scale, size=act_dim),
            -1.0,
            1.0,
        ).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(current_action)
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            randomize_base_torques(env)
            current_action[:] = 0.0
    env.close()

if __name__ == "__main__":
    main()
