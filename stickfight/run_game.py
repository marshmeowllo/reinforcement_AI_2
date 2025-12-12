"""Run a quick interactive StickmanFightEnv session.

Environment now uses motor-based control:
    Action space = 10 target angles for joints (normalized -1..1).

This script demonstrates both random play and lets you see the physics in action.

Usage:
    python run_game.py
    python run_game.py --random    # Both agents random
    python run_game.py --static    # Agent vs static opponent
    python run_game.py --model models/best/best_model.zip  # Use trained model
"""
from env import StickmanFightEnv
import numpy as np
import random
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--random', action='store_true', help='Both agents use random actions')
    parser.add_argument('--static', action='store_true', help='Opponent stays still')
    parser.add_argument('--model', type=str, default=None, help='Path to trained model (e.g., models/best/best_model.zip)')
    parser.add_argument('--steps', type=int, default=600, help='Steps per episode')
    parser.add_argument('--episodes', type=int, default=1, help='Number of episodes')
    args = parser.parse_args()
    
    # Load trained model if specified
    model = None
    if args.model:
        if not os.path.exists(args.model):
            print(f"Error: Model file not found: {args.model}")
            return
        
        try:
            from stable_baselines3 import PPO
            model = PPO.load(args.model)
            print(f"Loaded trained model: {args.model}")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Determine opponent policy
    if args.static:
        dummy_policy = "static"
        mode = "Trained Agent vs Static" if model else "Random Agent vs Static"
    elif args.random:
        dummy_policy = "random"
        mode = "Trained Agent vs Random" if model else "Both Random"
    else:
        dummy_policy = "random"
        mode = "Trained Agent vs Random" if model else "Smooth Random vs Random"
    
    print(f"{'='*60}")
    print(f"STICKMAN FIGHT DEMO")
    print(f"{'='*60}")
    print(f"Mode: {mode}")
    print(f"Episodes: {args.episodes}")
    print(f"Steps per episode: {args.steps}")
    print(f"{'='*60}\n")
    
    env = StickmanFightEnv(
        render_mode="human", 
        dummy_policy=dummy_policy, 
        role_swap=False
    )
    
    act_dim = env.action_space.shape[0]
    
    for episode in range(args.episodes):
        print(f"\n{'='*60}")
        print(f"Episode {episode + 1}/{args.episodes}")
        print(f"{'='*60}")
        
        obs, _ = env.reset()
        
        # Smooth random walk action (for random agent)
        current_action = np.zeros(act_dim, dtype=np.float32)
        noise_scale = 0.2
        
        total_reward = 0
        damage_dealt_total = 0
        damage_taken_total = 0
        
        for step in range(args.steps):
            # Generate action based on mode
            if model:
                # Use trained model
                current_action, _ = model.predict(obs, deterministic=False)
            elif args.random:
                # Pure random if --random flag
                current_action = np.random.uniform(-1.0, 1.0, size=act_dim).astype(np.float32)
            else:
                # Smooth random walk (more realistic)
                current_action = np.clip(
                    current_action + np.random.uniform(-noise_scale, noise_scale, size=act_dim),
                    -1.0,
                    1.0,
                ).astype(np.float32)
            
            obs, reward, terminated, truncated, info = env.step(current_action)
            env.render()
            
            # Track stats
            total_reward += reward
            damage_dealt_total += info.get('damage_dealt_self', 0)
            damage_taken_total += info.get('damage_taken_self', 0)
            
            # Print status every 100 steps
            if (step + 1) % 100 == 0:
                print(f"Step {step+1:3d}: "
                      f"Reward={total_reward:6.2f}, "
                      f"HP: Self={info['hp_a']:.1f} Opp={info['hp_b']:.1f}, "
                      f"Dmg: Dealt={damage_dealt_total:.1f} Taken={damage_taken_total:.1f}")
            
            if terminated or truncated:
                print(f"\n{'='*60}")
                print(f"Episode ended at step {step + 1}")
                print(f"{'='*60}")
                print(f"Final HP: Self={info['hp_a']:.1f}, Opponent={info['hp_b']:.1f}")
                print(f"Total Reward: {total_reward:.2f}")
                print(f"Damage Dealt: {damage_dealt_total:.2f}")
                print(f"Damage Taken: {damage_taken_total:.2f}")
                print(f"Net Damage: {damage_dealt_total - damage_taken_total:.2f}")
                
                if info['hp_a'] <= 0:
                    print("Result: OPPONENT WON (You were knocked out)")
                elif info['hp_b'] <= 0:
                    print("Result: YOU WON (Opponent knocked out)")
                else:
                    print("Result: Time expired")
                
                print(f"{'='*60}\n")
                break
        
        # Episode finished without termination
        if not (terminated or truncated):
            print(f"\nEpisode completed all {args.steps} steps")
            print(f"Final HP: Self={info['hp_a']:.1f}, Opponent={info['hp_b']:.1f}")
            print(f"Total Reward: {total_reward:.2f}")
    
    env.close()
    print(f"{'='*60}")


if __name__ == "__main__":
    main()