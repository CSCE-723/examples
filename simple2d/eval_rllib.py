"""
Evaluate a trained RLlib PPO model on Simple2DEnv and record videos of episodes.
"""
import os
import pathlib
import numpy as np
import ray
from ray.tune.registry import register_env
# from ray.rllib.env import BaseEnv
from gym_env.gym_env import Simple2DEnv
# import gymnasium as gym
import imageio
from ray.rllib.core.rl_module import RLModule
import torch

# Settings
checkpoint_path = pathlib.Path(
    ""
).absolute()
video_folder = "./rllib_logs/eval_videos"
num_episodes = 10
max_steps = 200

os.makedirs(video_folder, exist_ok=True)


def make_env(env_config={}):
    return Simple2DEnv(config={'render_mode': 'rgb_array'})


register_env("Simple2D", make_env)


# Create a single environment for evaluation (with rgb_array rendering)
def record_episode(rl_module, env, episode_idx, max_steps, video_folder):
    obs, _ = env.reset()
    frames = []
    done = False
    total_reward = 0
    step = 0
    while not done and step < max_steps:
        frame = env.render()
        if frame is not None:
            frames.append(frame)
        # RLlib expects a batch of obs (B=1)
        obs_batch = torch.from_numpy(obs).unsqueeze(0).float()
        model_outputs = rl_module.forward_inference({"obs": obs_batch})
        action_dist_params = model_outputs["action_dist_inputs"][0].numpy()
        # For continuous actions, take the mean (first element)
        # greedy_action = np.clip(
        #     action_dist_params[0:1],
        #     a_min=env.action_space.low[0],
        #     a_max=env.action_space.high[0],
        # )
        # For MultiDiscrete actions, take the argmax for each discrete action
        # action_dist_params shape: (sum of n_actions for each discrete dim,)
        # For each discrete action, logits are provided; take argmax per dim
        # RLlib docs: rllib-models.html#multidiscrete-action-spaces
        action_splits = env.action_space.nvec
        logits = action_dist_params
        greedy_action = []
        idx = 0
        for n in action_splits:
            # Take argmax over logits for this discrete action
            greedy_action.append(int(np.argmax(logits[idx:idx+n])))
            idx += n
        greedy_action = np.array(greedy_action)

        obs, reward, terminated, truncated, info = env.step(greedy_action)
        done = terminated or truncated
        total_reward += reward
        step += 1
    # Save video
    video_path = os.path.join(
        video_folder, f"rllib-simple2d-eval-rew{total_reward}-ep{episode_idx+1}.mp4"
    )
    if frames:
        imageio.mimsave(video_path, [np.array(f) for f in frames], fps=20)
    return total_reward, video_path


if __name__ == "__main__":
    ray.init(ignore_reinit_error=True)
    # Load the RLModule from the checkpoint
    rl_module = RLModule.from_checkpoint(
        pathlib.Path(checkpoint_path)
        / "learner_group"
        / "learner"
        / "rl_module"
        / "default_policy"
    )
    env = make_env()
    rewards = []
    for ep in range(num_episodes):
        total_reward, video_path = record_episode(rl_module, env, ep,
                                                  max_steps, video_folder)
        rewards.append(total_reward)
        print(f"Episode {ep+1}: reward={total_reward}, video={video_path}")
    print(
        f"Mean reward over {num_episodes} episodes: {np.mean(rewards):.2f} ± "
        f"{np.std(rewards):.2f}"
    )
    print(f"Evaluation videos saved to {video_folder}")
    ray.shutdown()
