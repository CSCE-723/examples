"""Custom callbacks to pass to RLlib for FlappyBird"""

import os
from typing import Dict, Any
import string
import imageio
import glob
import random
import numpy as np
import moviepy.editor as mpy
from datetime import datetime, timezone, timedelta
from ray.rllib.algorithms.algorithm import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class SaveReplayCallback(DefaultCallbacks):
    """Records a replay of the episode as a .mp4 or .gif file. Must have env render_mode='rgb_array'"""

    def __init__(self, replay_config = {}, legacy_callbacks_dict: Dict[str, Any] = None):
        """
        Args:
            replay_config -- a config dictionary with optional keys:
                'record_mode': 'off' or 'best'
                    default -- 'off'
                'replay_folder': pathlike to base folder of replay storage . 
                    sub folders for train and/or eval will be made based on 'eval_mode'. trial
                    folders will be created with timestamp
                    default -- './replays'
                'eval_mode': Bool whether .evaluation() trial type or not, for folder naming. 
                    default -- False
                'file_type': '.mp4' or '.gif'. '.gif' is not recommended due to file sizes
                    default -- '.mp4'
        """

        super().__init__(legacy_callbacks_dict)
        self.record_mode = replay_config.get('record_mode', 'off')
        self.base_folder = replay_config.get('replay_folder', './replays')
        self.file_type = replay_config.get('file_type', '.mp4')
        assert self.file_type in ['.gif', '.mp4'], 'Replay file_type must be .mp4 or .gif'
        self.eval_mode = replay_config.get('eval_mode', False)
        self.best_metric_max = -np.infty # init best reward for recording only best episodes
        if self.eval_mode:
            self.base_folder = os.path.join(self.base_folder, 'eval')
        else:
            self.base_folder = os.path.join(self.base_folder, 'train')
        
        if self.record_mode in ['off', 'OFF', 'Off', None, False]:
            self.record_mode = 'off'
            # print('REPLAY RECORDING SET TO OFF')
            return 
        
        if not os.path.exists(self.base_folder):
            os.makedirs(self.base_folder)

        timezone_offset = -5.0 # Eastern Standard Time (UTC−05:00)
        tzinfo = timezone(timedelta(hours=timezone_offset))
        now = datetime.now(tzinfo)
        now_str = now.strftime('%Y%m%d-%H%M%S')
        self.results_folder = os.path.join(self.base_folder, 'run_'+now_str)
        trial_list = sorted(glob.glob(os.path.join(self.base_folder, 'run_*')), key=lambda x: x.split('_')[-1])
        if trial_list:
            time = datetime.strptime(trial_list[-1].split('_')[-1], '%Y%m%d-%H%M%S').replace(tzinfo=tzinfo)
            diff = now - time
            if diff.total_seconds() < 30: 
                self.results_folder = trial_list[-1]

        if not os.path.exists(self.results_folder):
            os.makedirs(self.results_folder)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
        if self.record_mode == 'off': return # skip all this if not recording
        img = base_env.try_render()
        if not episode.user_data.get('img_list', []):
            episode.user_data['img_list'] = []
        episode.user_data['img_list'].append(img)
        
    
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        if self.record_mode == 'off': return # skip all this if not recording
        
        # get reward for this episode
        total_metric = round(episode.total_reward, ndigits=2)
        saved_replay_list = glob.glob(os.path.join(self.results_folder, '*'+self.file_type))
        saved_reward_list = [float(os.path.split(file)[-1].split('_')[0]) for file in saved_replay_list]
        if saved_replay_list:
            self.best_metric_max = max(self.best_metric_max, sorted(saved_reward_list)[-1])

        # Store this episode as the new best for this trial
        if (self.best_metric_max < total_metric) and (len(episode.user_data['img_list']) > 0):
            self.best_metric_max = total_metric
            best_img_list = episode.user_data['img_list']
            timezone_offset = -5.0 # Eastern Standard Time (UTC−05:00)
            tzinfo = timezone(timedelta(hours=timezone_offset))
            now = datetime.now(tzinfo)
            now_str = now.strftime('%Y%m%d-%H%M%S')
            filename = f"{self.best_metric_max:.2f}_reward_{now_str}_{id_generator()}"
            filepath = os.path.join(self.results_folder, filename)
            if self.file_type == 'gif':
                self.make_gif(filepath=filepath, img_list=best_img_list)
            else:
                self.make_mp4(filepath=filepath, img_list=best_img_list)
        
    def make_mp4(self, filepath, img_list):
        """.mp4 is recommended since .gifs can become very large"""
        clip = mpy.ImageSequenceClip(img_list, fps=30)
        clip.write_videofile(filepath+'.mp4', logger=None)

    def make_gif(self, filepath, img_list):
        """.gif not recommended, can become hundreds of MB very easily"""
        imageio.mimsave(filepath+'.gif', img_list)
    
class RestoreWeightsCallback(DefaultCallbacks):
    """Starts algorithm in tune trial with a previous set of policy_weights (this is buggy)"""
    def __init__(self, algorithm_weights = None, legacy_callbacks_dict: Dict[str, Any] = None):
        self.algorithm_weights = algorithm_weights
        super().__init__(legacy_callbacks_dict)

    def on_algorithm_init(self, *, algorithm: "Algorithm", **kwargs) -> None:
        algorithm.set_weights(self.algorithm_weights)

# TODO: Finish writing this callback
class FlapActionMetricCallback(DefaultCallbacks):
    """Keeps track of actions taken, saves a ratio (percentage-like) of actions that were flaps as a custom metric 'flap_ratio'"""
    def __init__(self, report_hist = False, legacy_callbacks_dict: Dict[str, Any] = None):
        self.report_hist = report_hist
        super().__init__(legacy_callbacks_dict)

    def on_episode_step(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy] | None = None, episode: Episode | EpisodeV2, env_index: int | None = None, **kwargs) -> None:
        # TODO: get the last action taken, make a custom metric
        # Hints: use a method from the 'episode' object in the args
        action = 0 # FIXME: get the action the agent took

        # Store this in an ongoing list in the 'user_data' of episode
        # this is a way to do a psuedo singleton thing on a python dictionary
        action_list = episode.user_data.get('actions', [])
        action_list.append(action)
        episode.user_data['actions'] = action_list

    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        action_list = episode.user_data.get('actions', [])
        if action_list:
            if self.report_hist:
                episode.hist_data['flap_hist'] = action_list
            flap_ratio = 0.0 # FIXME: calculate flap_ratio
            episode.custom_metrics['flap_ratio'] = flap_ratio 
        else:
            print('WARNING: There were no actions in the episode.user_data list!')
        
# TODO: Write this callback
class CustomScoreCallback(DefaultCallbacks):
    """Saves the end of episode "score" as a custom metric"""
    def on_episode_end(self, *, worker: RolloutWorker, base_env: BaseEnv, policies: Dict[PolicyID, Policy], episode: Episode | EpisodeV2 | Exception, env_index: int | None = None, **kwargs) -> None:
        # HINT: this method is all you need to save score as a custom metric
        pass # FIXME: write code in place of pass

# Helper functions (for recorder--student does not need this)
def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    """Generates a random id for a filename"""
    return ''.join(random.choice(chars) for _ in range(size))
