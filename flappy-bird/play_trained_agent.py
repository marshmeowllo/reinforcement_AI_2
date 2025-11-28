import os
from stable_baselines3 import DQN
from flappy_env import FlappyBirdEnv

def main():
    model_path = "models/flappy_dqn.zip"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please run train_agent.py first.")
        return

    # Create environment with human rendering
    env = FlappyBirdEnv(render_mode="human")
    
    # Load model
    model = DQN.load(model_path, env=env)

    print("Running trained agent... Press Ctrl+C to stop.")
    
    obs, _ = env.reset()
    try:
        while True:
            action, _states = model.predict(obs, deterministic=True)
            print(f"Action taken: {action}")
            obs, reward, terminated, truncated, info = env.step(action)
            
            if terminated or truncated:
                print(f"Game Over! Score: {info.get('score', 0)}")
                obs, _ = env.reset()
                
    except KeyboardInterrupt:
        print("Stopped by user.")
    finally:
        env.close()

if __name__ == "__main__":
    main()
