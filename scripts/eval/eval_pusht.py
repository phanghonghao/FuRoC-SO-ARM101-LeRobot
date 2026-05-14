"""PushT evaluation with Diffusion Policy — generates evaluation video."""
import sys
import os
import numpy as np
import torch
import imageio
from pathlib import Path

# Force CPU/GPU
DEVICE = os.environ.get("EVAL_DEVICE", "cuda:0")
CHECKPOINT = os.environ.get("CHECKPOINT_PATH", "/tmp/pusht_eval/pretrained_model")
OUTPUT = os.environ.get("OUTPUT_PATH", "/tmp/pusht_eval_video.mp4")
N_EPISODES = int(os.environ.get("N_EPISODES", "3"))
MAX_STEPS = int(os.environ.get("MAX_STEPS", "300"))

print(f"[eval_pusht] Device: {DEVICE}")
print(f"[eval_pusht] Checkpoint: {CHECKPOINT}")
print(f"[eval_pusht] Output: {OUTPUT}")
print(f"[eval_pusht] Episodes: {N_EPISODES}, Max steps: {MAX_STEPS}")

# Load policy
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy
print("[eval_pusht] Loading policy...")
policy = DiffusionPolicy.from_pretrained(CHECKPOINT)
policy.to(DEVICE)
policy.eval()
print(f"[eval_pusht] Policy loaded on {DEVICE}")

# Create environment
import gymnasium as gym
import gym_pusht
print("[eval_pusht] Creating PushT env...")
env = gym.make("gym_pusht/PushT-v0", obs_type="pixels_agent_pos")

all_frames = []

for ep in range(N_EPISODES):
    obs, info = env.reset()
    policy.reset()
    frames = []
    total_reward = 0.0

    for step in range(MAX_STEPS):
        # Render frame
        frame = env.render()
        if frame is not None:
            frames.append(frame.copy())

        # Prepare observation batch
        # obs["agent_pos"] shape: (2,) — agent position
        # obs["pixels"] shape: (480, 640, 3) — image
        agent_pos = obs["agent_pos"]
        pixels = obs["pixels"]

        state = torch.tensor(agent_pos, dtype=torch.float32).unsqueeze(0).to(DEVICE)  # (1, 2)
        image = torch.tensor(pixels, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(DEVICE)  # (1, 3, H, W)
        image = image / 255.0  # normalize to [0, 1]

        batch = {
            "observation.state": state,
            "observation.image": image,
            "observation.pixels": image,
        }

        with torch.no_grad():
            action = policy.select_action(batch)

        action_np = action.cpu().numpy().squeeze(0)
        obs, reward, terminated, truncated, info = env.step(action_np)
        total_reward += reward

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame.copy())
            break

    print(f"[eval_pusht] Episode {ep+1}/{N_EPISODES}: {len(frames)} frames, reward={total_reward:.3f}")
    all_frames.extend(frames)
    all_frames.append(np.zeros_like(frames[0]))  # black separator frame

env.close()

# Save video
if all_frames:
    # Remove last separator
    all_frames = all_frames[:-1]
    imageio.mimsave(OUTPUT, all_frames, fps=30)
    size_mb = os.path.getsize(OUTPUT) / 1024 / 1024
    print(f"[eval_pusht] Saved {OUTPUT} ({len(all_frames)} frames, {size_mb:.1f} MB)")
    print("[OK] Done!")
else:
    print("[eval_pusht] No frames captured!")
