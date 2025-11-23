"""Run a quick interactive StickmanFightEnv session.

Environment now uses motor-based control:
    Action space = 10 target angles for joints (normalized -1..1).

Reward rule:
    - Tie HP: -0.01 penalty.
    - Advantage: (hp_self - hp_opp) / MAX_HP + tiny standing bonus.

Usage:
    powershell> python run_game.py
"""
from stickfight.env import StickmanFightEnv
import numpy as np
import random

def main():
    # Use random dummy so the opponent keeps moving too
    env = StickmanFightEnv(render_mode="human", dummy_policy="random", role_swap=False)
    obs, _ = env.reset()
    # Boost hip (5,6) and knee (7,8) torques for stronger leg motion in this demo only
    TORQUE_MULT = 3.0
    leg_indices = [5, 6, 7, 8]
    for idx in leg_indices:
        env.agent_a.motors[idx].max_force *= TORQUE_MULT
        env.agent_b.motors[idx].max_force *= TORQUE_MULT
    # Optionally increase spine torque slightly for posture dynamics
    env.agent_a.motors[9].max_force *= 1.5
    env.agent_b.motors[9].max_force *= 1.5
    # No random movement
    act_dim = env.action_space.shape[0]
    current_action = np.zeros(act_dim, dtype=np.float32)
    for step in range(600):
        obs, reward, terminated, truncated, info = env.step(current_action.copy())
        env.render()
        if terminated or truncated:
            obs, _ = env.reset()
            current_action[:] = 0.0
    env.close()

if __name__ == "__main__":
    main()
