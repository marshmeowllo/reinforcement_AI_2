"""Live viewer that periodically loads newest training checkpoint and renders episodes.
Run separately from training:
    python live_view.py --interval 5
"""

import os
import time
import glob
import argparse
from stable_baselines3 import PPO
from stickfight.env import StickmanFightEnv


def latest_checkpoint(path: str):
    files = glob.glob(os.path.join(path, "*.zip"))
    if not files:
        return None
    return max(files, key=os.path.getmtime)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", default="models/checkpoints", help="Directory containing training checkpoints")
    p.add_argument("--interval", type=int, default=5, help="Seconds between polling for new checkpoint")
    p.add_argument("--episode-steps", type=int, default=400, help="Max steps per rendered episode")
    return p.parse_args()


def main():
    args = parse_args()
    env = StickmanFightEnv(render_mode="human", dummy_policy="random", role_swap=False)
    current = None
    model = None
    print(f"[Viewer] Watching directory: {args.checkpoint_dir}")
    try:
        while True:
            ckpt = latest_checkpoint(args.checkpoint_dir)
            if ckpt and ckpt != current:
                print(f"[Viewer] Loading {ckpt}")
                model = PPO.load(ckpt, device="cpu")
                current = ckpt
            if model:
                obs, _ = env.reset()
                for _ in range(args.episode_steps):
                    action, _ = model.predict(obs, deterministic=False)
                    #print(action)
                    obs, _, term, trunc, _ = env.step(action)
                    env.render()
                    if term or trunc:
                        break
            time.sleep(args.interval)
    finally:
        env.close()


if __name__ == "__main__":
    main()
