"""Custom callbacks to pass to stable_baselines3 for FlappyBird"""

from typing import Dict, Any
import numpy as np
import torch as th
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.logger import Video


# https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html
class FlapActionMetricCallback(BaseCallback):
    """
    Records a replay of the episode as a .mp4 or .gif file.
    Must have env render_mode='rgb_array'
    """

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        # Those variables will be accessible in the callback
        # (they are defined in the base class)
        # The RL model
        # self.model = None  # type: BaseAlgorithm
        # An alias for self.model.get_env(), the environment used for training
        # self.training_env # type: VecEnv
        # Number of time the callback was called
        # self.n_calls = 0  # type: int
        # num_timesteps = n_envs * n times env.step() was called
        # self.num_timesteps = 0  # type: int
        # local and global variables
        # self.locals = {}  # type: Dict[str, Any]
        # self.globals = {}  # type: Dict[str, Any]
        # The logger object, used to report things in the terminal
        # self.logger # type: stable_baselines3.common.logger.Logger
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        # self.parent = None  # type: Optional[BaseCallback]

    def _on_step(self) -> bool:
        """
        This method will be called by the model after each call to
          `env.step()`.

        For child callback (of an `EventCallback`), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        # This is a ndarray with the action from each env in the vecenv
        actions = self.training_env.actions
        if np.size(actions):
            # mean action chosen by n agents
            mean_action = sum(actions)/len(actions)
            # this keeps a running average of mean_action
            self.logger.record_mean('flap_ratio', mean_action)
        return True


class CustomScoreCallback(BaseCallback):
    """Saves the end of episode "score" as a custom metric"""
    def _on_step(self) -> bool:
        assert "dones" in self.locals, (
            "`dones` variable is not defined, please check your code next to "
            "`callback.on_step()`")
        assert "infos" in self.locals, (
            "`infos` variable is not defined, please check your code next to "
            "`callback.on_step()`")
        # ndarray of all done states (True/False episode over)
        dones = self.locals['dones']
        if any(dones):
            infos = self.locals['infos']
            # If env is done, record that env's score
            for i, done in enumerate(dones):
                if done:
                    self.logger.record_mean('score', infos[i]['score'])
        return True


#######################################################################
# No need to touch anything below this line
# These appear to be broken currently because of recent deprecations in moviepy
#######################################################################
class TBBestVideosCallback(BaseCallback):
    """
    Play n_episodes eval episodes and save best to Tensorboard
    Requires use with ``EvalCallback`` as:
        EvalCallback(callback_on_new_best=TensorboardBestVideosCallback())
    Only triggers on new best from EvalCallback
    Based on
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-videos
    """
    parent: EvalCallback

    def __init__(
            self,
            eval_env: gym.Env,
            n_eval_episodes: int = 1,
            deterministic: bool = True
            ):
        """
        Records a video of an agent's trajectory traversing ``eval_env`` and
          logs it to TensorBoard
        Must be called via EvalCallback(
         callback_on_new_best=TensorboardBestVideosCallback())

        :param eval_env: A gym environment from which the trajectory is
          recorded
        :param render_freq: Render the agent's trajectory every eval_freq call
          of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        assert self.parent is not None, (
            "``TensorboardBestVideosCallback`` callback must be used with an "
            "``EvalCallback``"
        )
        episode_screens = []
        screens = []

        def grab_screens(
                _locals: Dict[str, Any],
                _globals: Dict[str, Any]
                ) -> None:
            """
            Renders the environment in its current state, recording the screen
             in the captured `screens` list

            :param _locals: A dictionary containing all local variables of the
              callback's scope
            :param _globals: A dictionary containing all global variables of
              the callback's scope
            """
            assert self._eval_env.render_mode == "rgb_array", (
                "``TensorboardBestVideosCallback`` can only be used when "
                "`render_mode` of `eval_env`=='rgb_array'"
            )
            screen = self._eval_env.render()
            # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image convention
            screens.append(screen.transpose(2, 0, 1))
            if _locals['done']:
                episode_screens.append(screens.copy())
                screens.clear()

        episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
                return_episode_rewards=True,
            )
        for i, x in enumerate(episode_screens):
            print(len(x))
            print(episode_lengths[i])
        screens_out = episode_screens[np.argmax(episode_rewards)]
        self.logger.record(
                "trajectory/video_best",
                Video(th.ByteTensor([screens_out]), fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True


class TBVideoRecorderCallback(BaseCallback):
    """
    From
    https://stable-baselines3.readthedocs.io/en/master/guide/tensorboard.html#logging-videos
    """
    def __init__(
            self,
            eval_env: gym.Env,
            render_freq: int,
            n_eval_episodes: int = 1,
            deterministic: bool = True
            ):
        """
        Each `render_freq` env steps, records a video of an agent's trajectory
          traversing ``eval_env`` and logs it to TensorBoard

        :param eval_env: A gym environment from which the trajectory is
          recorded
        :param render_freq: Render the agent's trajectory every eval_freq call
          of the callback.
        :param n_eval_episodes: Number of episodes to render
        :param deterministic: Whether to use deterministic or stochastic policy
        """
        super().__init__()
        self._eval_env = eval_env
        self._render_freq = render_freq
        self._n_eval_episodes = n_eval_episodes
        self._deterministic = deterministic

    def _on_step(self) -> bool:
        if self.n_calls % self._render_freq == 0:
            screens = []

            def grab_screens(
                    _locals: Dict[str, Any],
                    _globals: Dict[str, Any]
                    ) -> None:
                """
                Renders the environment in its current state,
                  recording the screen in the captured `screens` list

                :param _locals: A dictionary containing all local variables of
                  the callback's scope
                :param _globals: A dictionary containing all global variables
                  of the callback's scope
                """
                screen = self._eval_env.render(mode="rgb_array")
                # PyTorch uses CxHxW vs HxWxC gym (and tensorflow) image
                #  convention
                screens.append(screen.transpose(2, 0, 1))

            evaluate_policy(
                self.model,
                self._eval_env,
                callback=grab_screens,
                n_eval_episodes=self._n_eval_episodes,
                deterministic=self._deterministic,
            )
            self.logger.record(
                "trajectory/video_latest",
                Video(th.ByteTensor([screens]), fps=30),
                exclude=("stdout", "log", "json", "csv"),
            )
        return True
