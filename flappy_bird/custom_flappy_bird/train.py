"""
A training script using tune.Tuner to handle a complex experiment. Uses a specified Gymnasium env and an Algorithm from RLlib
"""

from gym_env.custom_flappy_env import CustomFlappyBirdEnv
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms import AlgorithmConfig
import ray
import pathlib
from ray import tune, train
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from utils.rllib_callbacks import CustomScoreCallback, FlapActionMetricCallback, SaveReplayCallback

local_mode = False
training_iterations = 500 # max iterations before stopping - recommended
num_cpu = 10
num_gpus = 0
num_eval_workers = 1

driver_cpu = 1 # leave this alone
# How CPUs are spread
num_rollout_workers = num_cpu - driver_cpu - num_eval_workers

# # FIXME: Put the actual path you want
# ray_results = '/remote_home/EENG645A/Lab8/ray_results/'
ray_results = pathlib.Path(__file__).parent.parent.resolve().joinpath('ray_results')
replay_folder = pathlib.Path(__file__).parent.parent.resolve().joinpath('replays')

ray.shutdown() # Kill Ray incase it didn't stop cleanly in another run
ray.init(local_mode=local_mode) # set true for better debugging, but need be false for scaling up

# # We have to register the environment to be called as a trainable.  
# # This could be done in custom_flappy_bird/gym_env/__init__.py instead,
# #  since that file is run when importing its module... but doing this explicitly here to show this is needed
register_env("flappybird", CustomFlappyBirdEnv)
# # How RLlib will use this is equivalent to:
# register_env("flappybird", lambda env_config: CustomFlappyBirdEnv(env_config))
# # If you wanted to just pass args directly, you can also do this:
# register_env("flappybird", lambda _: CustomFlappyBirdEnv(render_mode='rgb_array', normalize_obs=True))

# Build all custom callbacks
callback_list = [CustomScoreCallback, 
                # This one needs a lambda to sneak our own argument in
                lambda: FlapActionMetricCallback(report_hist=False),
                lambda: SaveReplayCallback(replay_config={
                    'replay_folder': replay_folder,
                    'record_mode': 'off', # leave this off during training except to debug, slows it down a lot
                })
            ]

callbacks = make_multi_callbacks(callback_list)

# for config options see https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#ppo for example if ppo

# NOTE: _enable_new_api_stack and enable_connectors are set to false to force episode type to be EpisodeV1
# For some reason this version defaults to EpisodeV2 which is missing some methods we use here. Ray dev team is planning to add
# those methods back in with EpisodeV3. There's several ways to write callbacks either way but I digress, /rant.
# https://github.com/ray-project/ray/issues/37319
config = (  # 1. Configure the algorithm,
    AlgorithmConfig() # FIXME: put the actual config object for the algorithm you intend to use (Such as PPO or DQN)
    .environment('flappybird', env_config={'render_mode': 'rgb_array'})
    .experimental(_enable_new_api_stack=False)
    .rollouts(
        num_rollout_workers=num_rollout_workers,
        batch_mode='truncate_episodes',
        enable_connectors=False
        )
    .resources(num_gpus=num_gpus)
    .framework("tf2", eager_tracing=True)
    .training(
        # TODO: Put hyperparams here as needed. Look in the AlgorithmConfig object and child object for available params
        lr=2.5e-5,
        )
    .evaluation(evaluation_num_workers=num_eval_workers, evaluation_interval=10)
    .callbacks(callbacks)
    .reporting(keep_per_episode_custom_metrics=False) # decides whether custom metrics in Tensorboard are per episode or mean/min/max
)


tuner = tune.Tuner(
    "FIXME", # FIXME: Put the name that matches your alg name such as 'PPO' or 'DQN'
    run_config=train.RunConfig(
        name='Lab8FlappyBirdFIXME', # FIXME: Name this something reasonable
        local_dir=ray_results,
        stop={
            # "episode_reward_mean": 100, # another example of stopping criteria
            'training_iteration': training_iterations,
            },
        checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=True,
            checkpoint_score_attribute='episode_reward_mean',
            checkpoint_score_order='max',
            checkpoint_frequency=100,
            num_to_keep=5
            ),
    ),
    param_space=config
    )

results = tuner.fit()
df = results.get_dataframe()
print('Results DataFrame:\n', df)
print('Total steps trained: ', df['num_agent_steps_trained'])
print('Best checkpoint path:')
print()
print(results.get_best_result(metric='episode_reward_mean').checkpoint.path)
print()
