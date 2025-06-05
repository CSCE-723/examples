"""
Train with WANDB and Stable Baselines3 (SB3) using a custom environment.

A simple training script with sb3. For a more complete example similar to Ray,
see: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/train.py
"""

import pathlib
import os
# import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
import gymnasium  # noqa: F401
from gymnasium.envs.registration import register
from utils.utils import get_time_str, save_config
# from utils.sb3_callbacks import (  # noqa: F401
#     FlapActionMetricCallback,
#     CustomScoreCallback,
#     # TBBestVideosCallback,
#     # TBVideoRecorderCallback
# )
from stable_baselines3.common.callbacks import (  # noqa: F401
    EvalCallback,
    CheckpointCallback
)
from stable_baselines3.common.monitor import Monitor  # noqa: F401
from stable_baselines3.common.vec_env import (  # noqa: F401
    DummyVecEnv, VecVideoRecorder, SubprocVecEnv
)
import wandb
from wandb.integration.sb3 import WandbCallback

num_cpu = 10
resume_model = ''
# resume_config = '/home/developer/lab8-solution/models/PPO_20250306-120618/run_20250306-120618.json'
total_timesteps = 1e3
tensorboard_log = \
    pathlib.Path(__file__).parent.parent.resolve().joinpath('tblogs')
working_dir = pathlib.Path(__file__).parent.parent.resolve()
models_dir = working_dir.joinpath('models')
alg_name = 'PPO'  # just as a reminder later in config json

timestamp = get_time_str()
model_folder = os.path.join(models_dir, f'{alg_name}_{timestamp}')
vid_folder = os.path.join(working_dir, 'videos', f'{alg_name}_{timestamp}')

register(
     id="Simple2DEnv",
     entry_point="gym_env.gym_env:Simple2DEnv",
)

# This could be located in another file, or as a .json or .yml then
# imported/loaded.
# Recommend this as a simple way to keep track of your experiments.
config = dict(
    learning_rate=1e-4,
    gamma=0.99,
    gae_lambda=0.98,
    batch_size=128,
)
extra_config = dict(
    env_kwargs={
        'render_mode': 'rgb_array'
    },
    num_cpu=num_cpu,
    total_timesteps=total_timesteps,
    alg_name=alg_name,
    notes=''
)

save_config(
    config=config,
    timestamp=timestamp,
    folder=model_folder,
    extras=extra_config
)

run = wandb.init(
    project="Simple2D-test",
    config=config,
    sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=True,  # auto-upload the videos of agents playing the game
    save_code=True,  # optional
)

# Parallel environments
env = make_vec_env(
    "Simple2DEnv",
    n_envs=num_cpu,
    env_kwargs=extra_config['env_kwargs'],
    monitor_dir=os.path.join(model_folder, 'monitor'),
    # vec_env_cls=SubprocVecEnv,
)
# Can be used to enable period recording during training
# env = VecVideoRecorder(
#     env,
#     f"videos/{run.id}",
#     record_video_trigger=lambda x: x % 20000 == 0,
#     video_length=200,
# )

# Env to do periodic evaluation
# eval_env = make_vec_env(
#     "CustomFlappyBirdEnv",
#     n_envs=1,
#     env_kwargs={'render_mode': 'rgb_array'},
#     monitor_dir=os.path.join(model_folder, 'eval_monitor')
# )
# eval_env = VecVideoRecorder(
#     eval_env,
#     vid_folder,
#     record_video_trigger=lambda i:True,
#     video_length=2000,
# )

if resume_model:
    alg = PPO.load(
        resume_model,
        env=env,
        device='cpu'
    )

else:
    alg = PPO(
        'MlpPolicy',
        env=env,
        device='cpu',
        verbose=1,
        tensorboard_log=model_folder,
        **config,
    )

alg.learn(
    total_timesteps=total_timesteps,
    progress_bar=True,
    tb_log_name=f'{alg_name}_{timestamp}',
    callback=[
        # FlapActionMetricCallback(),
        # CustomScoreCallback(),
        CheckpointCallback(
            save_freq=1000,
            save_path=model_folder
        ),
        WandbCallback(
            model_save_path=model_folder,
            model_save_freq=1000,
            verbose=2
        ),
        # Turn this on to use eval env 
        # EvalCallback(
        #     eval_env=eval_env,
        #     n_eval_episodes=5,
        #     eval_freq=100000,
        #     log_path=tensorboard_log,
        #     best_model_save_path=model_folder,
        #     deterministic=True,
        #     render=False,
        #     verbose=1,
        # ),
    ]
)

alg.save(os.path.join(model_folder, 'end_model.zip'))

run.finish()
