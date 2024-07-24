"""
Lab8 Evaluation Script

This script takes whatever checkpoint path you put on the variable 'checkpoint', then loads it and runs it for 'eval_duration' number of episodes

This is useful to evaluate how your model actually does post training according to your design of experiments. 
Assuming the performance is whatever the end of a mean metric graph is does not tell the full story, because RL policies can often
be inconsistent or fail in some important edge cases while performing well in others.

If this does not work as expected, try using the fallback script eval_manual.py
"""

import pathlib
import os
import time
import sys
import pprint
import argparse
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import make_multi_callbacks
from gym_env.custom_flappy_env import CustomFlappyBirdEnv
from utils.rllib_callbacks import CustomScoreCallback, FlapActionMetricCallback, SaveReplayCallback

# Fix tensorflow bug when using tf2 framework with Algorithm.from_checkpoint()
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

# FIXME: use the actual checkpoint, up to the folder depth as in this example
# This can be copy/paste from the terminal if "best checkpoint" is printed,
# Or in vscode you may right click on a folder and choose "copy path" to get the absolute path
checkpoint = '/root/lab8-template/ray_results/Lab8FlappyBirdFIXME/PPO_flappybird_5fd0b_00000_0_2024-02-21_21-15-44/checkpoint_000000'
replay_folder = pathlib.Path(__file__).parent.parent.resolve().joinpath('replays')
evaluation_duration = 20
evaluation_num_workers = 5

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

def evaluate(checkpoint: str | os.PathLike, replay_folder: str | os.PathLike, evaluation_duration: int = 20, evaluation_num_workers: int = 5):
    """Runs a model evaluation from checkpoint for a duration with num workers and saves replays to folder"""

    # register_env("flappybird", lambda env_config: CustomFlappyBirdEnv(env_config))
    register_env("flappybird", CustomFlappyBirdEnv)

    # Get the old config from the checkpoint
    old_alg = Algorithm.from_checkpoint(checkpoint=checkpoint)
    old_config = old_alg.get_config()
    config = old_config.copy(copy_frozen=False) # make an unfrozen copy

    # Update config for evaluation only run
    config_update = {
        'env_config': {'render_mode': 'rgb_array'},
        'evaluation_config': {
            'evaluation_interval': 1,
            'evaluation_duration_unit': 'episodes',
            'evaluation_duration': evaluation_duration,
            'evaluation_num_workers': evaluation_num_workers,
        },
        'num_rollout_workers': 0,
        'callbacks': make_multi_callbacks([
            CustomScoreCallback, 
            # This one needs a lambda to sneak our own argument in
            lambda: FlapActionMetricCallback(report_hist=False),
            lambda: SaveReplayCallback(replay_config={
                'replay_folder': replay_folder,
                'record_mode': 'best',
                'eval_mode': True,
            })
        ]),
        # 'explore': False, # NOTE: DO NOT turn explore off with policy algs like PPO
    }
    config.update_from_dict(config_update)

    # build new alg
    alg = config.build()

    # # restore the policy and training history
    alg.restore(checkpoint_path=checkpoint)

    # Run the evaluation
    tic = time.perf_counter()
    eval_results = alg.evaluate(duration_fn=eval_duration_fn_args(eval_duration=evaluation_duration, tic=tic))

    # Report how it went
    print('Custom metrics: ')
    pprint.pprint(eval_results['evaluation']['custom_metrics'])
    print('Reward mean: ', eval_results['evaluation']['episode_reward_mean'])
    print('Score mean: ', eval_results['evaluation']['custom_metrics']['score_mean'])
    seconds = time.perf_counter() - tic
    mm, ss = divmod(seconds, 60)
    hh, mm = divmod(mm, 60)
    print(f'Total time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')

if __name__ == '__main__':
    evaluate(checkpoint=checkpoint, replay_folder=replay_folder, evaluation_duration=evaluation_duration, evaluation_num_workers=evaluation_num_workers)
