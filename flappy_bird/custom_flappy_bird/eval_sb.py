"""Simple script to evaluate on a sb3 saved policy"""

# import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from gymnasium.envs.registration import register
from utils.utils import get_time_str
# from utils.sb3_callbacks import FlapActionMetricCallback
from utils.wrappers import RecordBestVideo
from stable_baselines3.common.vec_env import VecVideoRecorder

# Paste the path to your model zip file here
model_file = ""

num_cpu = 10
n_eval_episodes = 20

register(
     id="CustomFlappyBirdEnv",
     entry_point="gym_env.custom_flappy_env:CustomFlappyBirdEnv",
)

vec_env = make_vec_env(
    "CustomFlappyBirdEnv",
    n_envs=num_cpu,
    env_kwargs={'render_mode': "rgb_array"},
    monitor_dir='./monitor',
    wrapper_class=RecordBestVideo,
    wrapper_kwargs={
          'video_folder': f'./replays/eval/run_{get_time_str()}',
          'name_prefix': "sb3-flappy",
          'record_mode': "best",
          'reward_in_name': True,
          'second_metric': 'score',
    }
    )
# This one can be used if you need to break up the video into smaller chunks
# It is possible on very long videos to run out of memory and crash the process
# vec_env = VecVideoRecorder(
#     vec_env,
#     f"videos/",
#     record_video_trigger=True,
#     video_length=20_000,
# )

alg = PPO.load(
    model_file,
    env=vec_env,
    # device='cpu'
    )

mean_reward, std_reward = evaluate_policy(
    alg,
    env=vec_env,
    n_eval_episodes=n_eval_episodes,
    )

print('Mean reward: ', mean_reward)
print('Std reward: ', std_reward)
