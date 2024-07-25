"""
A training script using tune.Tuner to handle a complex experiment. Uses a specified Gymnasium env and an Algorithm from RLlib
"""

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import AlgorithmConfig
import ray
import pathlib
from ray import tune, train
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from gym_env.gym_env import Simple2DEnv
from ray.air.integrations.wandb import WandbLoggerCallback, setup_wandb

local_mode = False
training_iterations = 30 # max iterations before stopping - recommended
num_cpu = 10
num_gpus = 0
num_eval_workers = 1
driver_cpu = 1 # leave this alone
# How CPUs are spread
num_rollout_workers = num_cpu - driver_cpu - num_eval_workers

ray_results = pathlib.Path(__file__).parent.parent.resolve().joinpath('ray_results')
replay_folder = pathlib.Path(__file__).parent.parent.resolve().joinpath('replays')

ray.shutdown() # Kill Ray incase it didn't stop cleanly in another run
ray.init(local_mode=local_mode) # set true for better debugging, but need be false for scaling up

config = (  # 1. Configure the algorithm,
    PPOConfig() # FIXME: put the actual config object for the algorithm you intend to use (Such as PPO or DQN)
    .environment(Simple2DEnv, env_config={'render_mode': 'rgb_array'})
    .experimental(_enable_new_api_stack=False)
    .env_runners(num_env_runners=num_rollout_workers, batch_mode='truncate_episodes')
    .framework("tf2", eager_tracing=True)
    .training(
        # TODO: Put hyperparams here as needed. Look in the AlgorithmConfig object and child object for available params
        # lr=2.5e-5,
        )
    .evaluation(evaluation_num_workers=num_eval_workers, evaluation_interval=10)
)


tuner = tune.Tuner(
    "PPO", # FIXME: Put the name that matches your alg name such as 'PPO' or 'DQN'
    run_config=train.RunConfig(
        name='2DGymExample', # FIXME: Name this something reasonable
        storage_path=ray_results,
        stop={
            # "episode_reward_mean": 100, # another example of stopping criteria
            'training_iteration': training_iterations,
            },
        checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_score_attribute='episode_reward_mean',
            checkpoint_score_order='max',
            checkpoint_frequency=10,
            num_to_keep=5
            ),
        callbacks=[WandbLoggerCallback(project='2DGymTest')]
    ),
    param_space=config
    )

results = tuner.fit()
