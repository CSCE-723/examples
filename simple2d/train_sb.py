"""
A simple training script with sb3. For a more complete example similar to Ray,
see: https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/rl_zoo3/train.py
"""

import pathlib
import os

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from gymnasium.envs.registration import register
from utils.utils import get_time_str, save_config
# from utils.sb3_callbacks import (  # noqa: F401
#     FlapActionMetricCallback,
#     CustomScoreCallback,
#     # TBBestVideosCallback,
#     # TBVideoRecorderCallback
# )
from stable_baselines3.common.callbacks import EvalCallback

num_cpu = 10
total_timesteps = 1e3
tensorboard_log = \
    pathlib.Path(__file__).parent.parent.resolve().joinpath('tblogs')
models_dir = pathlib.Path(__file__).parent.parent.resolve().joinpath('models')
alg_name = 'PPO'  # just as a reminder later in config json

timestamp = get_time_str()
model_folder = os.path.join(models_dir, f'{alg_name}_{timestamp}')

# This could be located in another file, or as a .json or .yml then
# imported/loaded.
# Recommend this as a simple way to keep track of your experiments.
config = {
    'alg_name': alg_name,
    'env_kwargs': {
        'render_mode': 'rgb_array'
    },
    'learning_rate': 2.5e-5,
}

save_config(
    config=config,
    timestamp=timestamp,
    folder=model_folder
)

register(
     id="Simple2DEnv",
     entry_point="gym_env.gym_env:Simple2DEnv",
)

# Parallel environments
vec_env = make_vec_env(
    "Simple2DEnv",
    n_envs=num_cpu,
    env_kwargs=config['env_kwargs'],
    monitor_dir=os.path.join(model_folder, 'monitor')
)

eval_env = make_vec_env(
    "Simple2DEnv",
    n_envs=1,
    env_kwargs=config.get(
        'eval_kwargs',
        {'render_mode': 'rgb_array'}
    ),
    monitor_dir=os.path.join(model_folder, 'eval_monitor')
)

alg = PPO(
    "MlpPolicy",
    vec_env,
    learning_rate=config['learning_rate'],
    verbose=1,
    tensorboard_log=tensorboard_log
)

alg.learn(
    total_timesteps=total_timesteps,
    progress_bar=True,
    tb_log_name=f'{alg_name}_{timestamp}',
    callback=[
        # FlapActionMetricCallback(),
        # CustomScoreCallback(),
        EvalCallback(
            eval_env=eval_env,
            # callback_on_new_best=TBBestVideosCallback(
            #     eval_env=eval_env,
            #     n_eval_episodes=5,
            #     deterministic=True
            #     ),
            n_eval_episodes=5,
            eval_freq=100000,
            log_path=tensorboard_log,
            best_model_save_path=model_folder,
            deterministic=True,
            render=False,
            verbose=1,
        ),
        # TBVideoRecorderCallback(
        #     eval_env=eval_env,
        #     render_freq=10000,
        #     n_eval_episodes=5,
        #     deterministic=True
        # )
    ]
)

alg.save(os.path.join(model_folder, 'model.zip'))

# # del alg # remove to demonstrate saving and loading

# alg = PPO.load(os.path.join(model_folder, 'model.zip'))

# # If you have X forwarding setup, you can render in human mode
# # to launch a pygame window
# obs = vec_env.reset()
# # vec_env.render_mode = "human"
# # vec_env.unwrapped.render_mode = "human"
# while True:
#     action, _states = alg.predict(obs)
#     obs, rewards, dones, info = vec_env.step(action)
#     vec_env.render()
