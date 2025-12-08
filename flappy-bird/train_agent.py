import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from flappy_env import FlappyBirdEnv

def main():
    # Create directories
    models_dir = "models"
    log_dir = "logs"
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Initialize environment
    env = FlappyBirdEnv(render_mode=None) # No rendering during training for speed
    env = Monitor(env, log_dir)

    # Initialize Agent
    # MlpPolicy is suitable for vector observations
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
                learning_rate=1e-4, buffer_size=50000, learning_starts=1000, 
                target_update_interval=1000, train_freq=4, gradient_steps=1,
                exploration_fraction=0.1, exploration_final_eps=0.02)

    print("Starting training...")
    # Train the agent
    # 100,000 timesteps should be enough to see some progress
    total_timesteps = 200000 
    model.learn(total_timesteps=total_timesteps)
    print("Training finished.")    # Save the model
    model_path = os.path.join(models_dir, "flappy_dqn")
    model.save(model_path)
    print(f"Model saved to {model_path}.zip")

    # Close training env
    env.close()

    # --- Demonstration ---
    print("Starting demonstration...")
    env = FlappyBirdEnv(render_mode="human")
    
    # Load the trained model
    model = DQN.load(model_path, env=env)

    obs, _ = env.reset()
    done = False
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        env.render()
        
        if terminated or truncated:
            obs, _ = env.reset()

if __name__ == "__main__":
    main()
