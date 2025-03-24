"""Simple script to evaluate on a sb3 saved policy"""

import pathlib
import os
import glob

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from utils.utils import get_time_str, load_config
# from utils.sb3_callbacks import FlapActionMetricCallback
# from utils.wrappers import RecordBestVideo
from stable_baselines3.common.vec_env import VecVideoRecorder
from gym_env.gym_env import Simple2DEnv

# models_dir = pathlib.Path(__file__).parent.parent.resolve().joinpath('models')
# run = 'models/PPO_20250324-085014/PPO_20250324-085014_1'

run_folder = '/home/developer/CSCE723-examples/models/PPO_20250324-085014'
model = os.path.join(run_folder, 'model.zip')

config_path = glob.glob(os.path.join(run_folder, '*.json'))[0]
config = load_config(config_path)

num_cpu = 1
n_eval_episodes = 2

register(
     id="Simple2DEnv",
     entry_point="gym_env.gym_env:Simple2DEnv",
)

env = lambda **kwargs: Simple2DEnv(render_mode='rgb_array')

vec_env = make_vec_env(
    # "Simple2DEnv",
    env,
    n_envs=num_cpu,
    env_kwargs=dict(render_mode='rgb_array'),
    monitor_dir='./monitor',
    # wrapper_class=RecordBestVideo,
    # wrapper_kwargs={
    #       'video_folder': f'./replays/eval/run_{get_time_str()}',
    #       'name_prefix': "sb3-flappy",
    #       'record_mode': "best",
    #       'reward_in_name': True,
    #       'second_metric': 'score',
    # }
    )
# This one can be used if you need to break up the video into smaller chunks
# It is possible on very long videos to run out of memory and crash the process
vec_env = VecVideoRecorder(
    vec_env,
    f"videos/",
    record_video_trigger=lambda _: True,
    video_length=200,
)

alg = PPO.load(model, env=vec_env, device='cpu')

mean_reward, std_reward = evaluate_policy(
    alg,
    env=vec_env,
    n_eval_episodes=n_eval_episodes,
)
print('Mean reward: ', mean_reward)
print('Std reward: ', std_reward)

vec_env.close()  # Add this to properly clean up the environment and video recorder
