"""
Evaluation script - manual config fallback

This is a 'manual' fallback method for running an evaluation, should the Ray config loader not work as expected.
Instead of loading the algorithm from a checkpoint then updating the config for evaluation, a new algorithm config
is created and then the checkpoint state is loaded into it.
The config here should match or be compatible with what was trained, or else the results will be garbage-in -> garbage-out.
"""

import pathlib
import time
import sys
import pprint
from ray.tune import register_env
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.algorithms.dqn import DQNConfig
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from gym_env.custom_flappy_env import CustomFlappyBirdEnv
from utils.rllib_callbacks import CustomScoreCallback, FlapActionMetricCallback, SaveReplayCallback

# FIXME: use the actual checkpoint, up to the folder depth as in this example
checkpoint = '/root/eeng645-rllab/ray_results/Lab8FlappyBird/PPO_flappybird_b3ba3_00000_0_2024-02-17_21-44-18/checkpoint_000009'
replay_folder = pathlib.Path(__file__).parent.resolve().joinpath('replays')
eval_duration = 10

def progressBar(count_value, total, suffix=''):
    """Makes a progress bar in the terminal according to count_value out of total steps"""
    bar_length = 100
    filled_up_Length = int(round(bar_length* count_value / float(total)))
    percentage = round(100.0 * count_value/float(total),1)
    bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
    sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
    sys.stdout.flush()

def eval_duration_fn_args(eval_duration, tic):
    """Allows total episodes :eval_duration: and start time :tic: to be used in eval_duration_fn"""
    def eval_duration_fn(num_units_done):
        """Makes a progress bar during alg.evaluate()"""
        print(f'Episodes evaluated: {num_units_done}/{eval_duration}')
        seconds = time.perf_counter() - tic
        mm, ss = divmod(seconds, 60)
        hh, mm = divmod(mm, 60)
        print(f'Time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')
        progressBar(num_units_done, eval_duration)
        # progressBar.update(n=num_units_done)
        return eval_duration-num_units_done
    return eval_duration_fn

# One way to register the env
register_env("flappybird", lambda _: CustomFlappyBirdEnv(render_mode='rgb_array', normalize_obs=True))

# NOTE: The important bits need to be configured the same as in training 
# Having a different config than was used in training will cause garbage_in -> garbage_out effect
# So it is important to keep track of your configs! You may want to make a system to log/tag/save them to avoid confusing yourself later
config = (  
    AlgorithmConfig() # replace with actual algorithm config object
    .environment('flappybird')
    .experimental(_enable_new_api_stack=False)
    # No rollout workers (training workers) since eval only
    .rollouts(num_rollout_workers=0, enable_connectors=False)
    .framework("tf2", eager_tracing=True)
    .training(model={"fcnet_hiddens": [256, 256], 'fcnet_activation': 'relu'})
    # # .evaluation() is where it needs to be changed
    .evaluation(
        evaluation_interval=1, 
        evaluation_duration_unit='episodes',
        evaluation_duration=eval_duration,
        evaluation_num_workers=5
    )
    # # you may also change callbacks pretty safely 
    .callbacks(make_multi_callbacks([
        CustomScoreCallback, 
        # This one needs a lambda to sneak our own argument in
        lambda: FlapActionMetricCallback(report_hist=False),
        lambda: SaveReplayCallback(replay_config={
            'replay_folder': replay_folder,
            'record_mode': 'best',
            'eval_mode': True,
        })
        ]))
    .reporting(keep_per_episode_custom_metrics=False) # decides whether custom metrics in Tensorboard are per episode or mean/min/max
)

alg = config.build()
alg.restore(checkpoint_path=checkpoint)

# Run the evaluation
tic = time.perf_counter()
eval_results = alg.evaluate(duration_fn=eval_duration_fn_args(eval_duration=eval_duration, tic=tic))
pprint.pprint(eval_results['evaluation']['custom_metrics'])
print('Score: ', eval_results['evaluation']['custom_metrics']['score_mean'])
seconds = time.perf_counter() - tic
mm, ss = divmod(seconds, 60)
hh, mm = divmod(mm, 60)
print(f'Total time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')
