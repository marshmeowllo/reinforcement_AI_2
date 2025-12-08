import os
import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from flappy_env import FlappyBirdEnv

# Create directories
models_dir = "models"
log_dir = "logs"
os.makedirs(models_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

def make_env():
    """Create and wrap the environment"""
    def _init():
        env = FlappyBirdEnv(render_mode=None)
        env = Monitor(env)
        return env
    return _init

def main():
    # Initialize environment
    env = DummyVecEnv([make_env()])
    eval_env = DummyVecEnv([make_env()])

    # Initialize Agent
    # MlpPolicy is suitable for vector observations
    model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=log_dir, 
                learning_rate=1e-4, buffer_size=100_000, learning_starts=5_000, batch_size=64,
                target_update_interval=1_000, tau=1.0, train_freq=4, gradient_steps=1,
                exploration_initial_eps=1.0, exploration_fraction=0.3, exploration_final_eps=0.01,
                gamma=0.99)
    
    checkpoint_callback = CheckpointCallback(
        save_freq=100_000,
        save_path=models_dir,
        name_prefix="flappy_dqn_checkpoint"
    )
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=models_dir,
        log_path=log_dir,
        eval_freq=20_000,
        deterministic=True,
        render=False,
    )

    # --- Training ---
    print("Starting training...")
    total_timesteps = 200_000 
    try:
        model.learn(
            total_timesteps=total_timesteps,
            callback=[checkpoint_callback, eval_callback],
            progress_bar=True
        )
        print("Training finished successfully!")
    except KeyboardInterrupt:
        print("Training interrupted by user.")

    # Save the final model
    model_path = os.path.join(models_dir, "flappy_dqn_final")
    model.save(model_path)
    print(f"Final model saved to {model_path}.zip")

    # Close training env
    env.close()
    eval_env.close()

    # --- Demonstration ---
    print("\nStarting demonstration with best model...")
    env = FlappyBirdEnv(render_mode="human")
    
    try:
        model = DQN.load(os.path.join(models_dir, "best_model"), env=env)
        print("Loaded best model from evaluation")
    except:
        model = DQN.load(model_path, env=env)
        print("Loaded final model")

    obs, _ = env.reset()
    episode_reward = 0
    episode_count = 0
    done = False
    
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            env.render()
            
            if terminated or truncated:
                episode_count += 1
                print(f"Episode {episode_count}: Score = {info['score']}, Reward = {episode_reward:.2f}")
                episode_reward = 0
                obs, _ = env.reset()
    except KeyboardInterrupt:
        print("\nDemonstration stopped by user")
    finally:
        env.close()

if __name__ == "__main__":
    main()
