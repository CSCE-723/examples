"""
Evaluate a trained SB3 PPO model on TankEnv and record videos of episodes.
"""
import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.evaluation import evaluate_policy
from gym_env.gym_env import Simple2DEnv

# Settings
model_path = "/workspace/sb3_logs/ppo_simple2d_final.zip"
video_folder = "./sb3_logs/eval_videos"
num_episodes = 1
max_steps = 200

os.makedirs(video_folder, exist_ok=True)


# Create a single environment for evaluation (with rgb_array rendering)
def make_env():
    def _init():
        return Simple2DEnv(config={'render_mode': 'rgb_array'})
    return _init


env = DummyVecEnv([make_env()])
env = VecVideoRecorder(
    env,
    video_folder=video_folder,
    record_video_trigger=lambda x: True,  # record every episode
    video_length=max_steps,
    name_prefix="ppo-simple2d-eval",
)

# Load the trained model
model = PPO.load(model_path)

# Evaluate using SB3's evaluate_policy (this will step through episodes)
mean_reward, std_reward = evaluate_policy(
    model,
    env,
    n_eval_episodes=num_episodes,
    deterministic=True,
    render=False,  # Rendering is handled by VecVideoRecorder
    return_episode_rewards=False,
)

# Print mean and std reward (break up long line for PEP8)
print(
    f"Mean reward over {num_episodes} episodes: "
    f"{mean_reward:.2f} ± {std_reward:.2f}"
)
print(f"Evaluation videos saved to {video_folder}")

env.close()
