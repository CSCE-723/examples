"""
This is a Command Line Interface (CLI) to evaluate a checkpoint with passed args.
May also be run as a script when changing default_checkpoint

For more info, in terminal from the workspace dir, type:
    python eval.py -h
"""

import time
import pathlib
import pprint
from ray.tune import register_env
from ray.rllib.algorithms.algorithm import Algorithm
from gym_env.gym_env import Simple2DEnv
from gymnasium.wrappers.record_video import RecordVideo

# Fix tensorflow bug when using tf2 framework with Algorithm.from_checkpoint()
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()
tf1.enable_eager_execution()

# # Put the checkpoint to run episodes on here, as well as number of workers and episodes to use. Looks something like this.
checkpoint = '/workspaces/CSCE723-examples/ray_results/2DGymExample/PPO_Simple2DEnv_ed152_00000_0_2024-07-25_20-10-23/checkpoint_000002'
replay_folder = pathlib.Path(__file__).parent.joinpath('replays')
evaluation_duration = 10
evaluation_num_workers = 1

env_creator = lambda env_config: RecordVideo(Simple2DEnv(env_config), video_folder=replay_folder)
register_env('Simple2DEnv', env_creator)

# Get the old config from the checkpoint
old_alg = Algorithm.from_checkpoint(checkpoint=checkpoint)
old_config = old_alg.get_config()
config = old_config.copy(copy_frozen=False) # make an unfrozen copy

# Update config for evaluation only run
config_update = {
    'env': 'Simple2DEnv',
    'env_config': {
        # env updates here
        },
    'evaluation_config': {
        'evaluation_interval': 1,
        'evaluation_duration_unit': 'episodes',
        'evaluation_duration': evaluation_duration,
        'evaluation_num_workers': evaluation_num_workers,
    },
    'num_rollout_workers': 0,
    # 'explore': False, # NOTE: DO NOT turn explore off with policy algs like PPO
}

# Update config with dictionary
config.update_from_dict(config_update)

# Build new alg
alg: Algorithm = config.build()

# Restore the policy and training history
alg.restore(checkpoint_path=checkpoint)

# Run the evaluation
tic = time.perf_counter()
eval_results = alg.evaluate()

# Report how it went
print('Reward mean: ', eval_results['env_runners']['episode_reward_mean'])
print('Reward per episode: ')
pprint.pprint([x for x in enumerate(eval_results['env_runners']['hist_stats']['episode_reward'])])
# pprint.pprint(eval_results)
seconds = time.perf_counter() - tic
mm, ss = divmod(seconds, 60)
hh, mm = divmod(mm, 60)
print(f'Total time elapsed: {int(hh)}h {int(mm)}m {int(ss)}s')
print('Evaluation duration: ', evaluation_duration)
print('Replay folder: ',replay_folder)
