"""Custom gymnasium env that inherits from FlappyBirdEnv"""

from typing import Dict, Tuple
import numpy as np
# import gymnasium as gym
from numpy import ndarray
from flappy_bird_gymnasium import FlappyBirdEnv
from flappy_bird_gymnasium.envs.flappy_bird_env import Actions
import pygame
from flappy_bird_gymnasium.envs.constants import (
    PIPE_HEIGHT,
    PIPE_WIDTH,
    PLAYER_MAX_VEL_Y
)

class CustomFlappyBirdEnv(FlappyBirdEnv):
    """
    Inherits a FlappyBirdEnv 0.4.0 for customization
    Uses the 0.4.0 version on pip:
    https://github.com/markub3327/flappy-bird-gymnasium/tree/v0.4.0
    """

    def __init__(
            self,
            env_config={},
            screen_size: Tuple[int] = (288, 512),
            audio_on: bool = False,
            normalize_obs: bool = True,
            use_lidar: bool = False,
            pipe_gap: int = 100,
            bird_color: str = "yellow",
            pipe_color: str = "green",
            render_mode: str | None = None,
            background: str | None = "day",
            score_limit: int | None = None,
            debug: bool = False
            ) -> None:
        """
        env_config dict may be used to overwrite arguments.
        use_lidar has its default changed to False.
        """

        # This enables env_configs passed through
        # if using ray to overwrite  __init__ args
        screen_size = env_config.get('screen_size', screen_size)
        audio_on = env_config.get('audio_on', audio_on)
        normalize_obs = env_config.get('normalize_obs', normalize_obs)
        use_lidar = env_config.get('use_lidar', use_lidar)
        pipe_gap = env_config.get('pipe_gap', pipe_gap)
        bird_color = env_config.get('bird_color', bird_color)
        pipe_color = env_config.get('pipe_color', pipe_color)
        render_mode = env_config.get('render_mode', render_mode)
        background = env_config.get('background', background)
        score_limit = env_config.get('score_limit', score_limit)
        debug = env_config.get('debug', debug)
        super().__init__(
            screen_size,
            audio_on,
            normalize_obs,
            use_lidar,
            pipe_gap,
            bird_color,
            pipe_color,
            render_mode,
            background,
            score_limit,
            debug
        )
        self.custom_reward_weight = env_config.get('custom_reward_weight', 0.5)

    def step(
            self,
            action: Actions | int
            ) -> Tuple[ndarray | float | bool | Dict]:

        obs, reward, terminal, truncated, info = super().step(action)

        reward = reward-self.get_custom_reward()

        return (
            obs,
            reward,
            terminal,
            truncated,
            info,
        )

    def reset(
            self,
            seed=None,
            options=None
            ) -> Tuple[ndarray | Dict]:
        obs, info = super().reset(seed, options)

        # reset your changes to env here as needed

        return obs, info

    # TODO: Write code here that makes it so the score is printed on the
    # screen in videos. HINT: find a parent method to override.
    def render(self) -> None:
        """
        Renders the next frame.
        Modified to print score in videos.
        """
        if self.render_mode == "rgb_array":
            self._draw_surface(show_score=True, show_rays=False)
            # Flip the image to retrieve a correct aspect
            return np.transpose(
                pygame.surfarray.array3d(self._surface), axes=(1, 0, 2))
        else:
            self._draw_surface(show_score=True, show_rays=self._use_lidar)
            if self._display is None:
                self._make_display()

            self._update_display()
            self._fps_clock.tick(self.metadata["render_fps"])

    # Everything below here is getting fancy and not required for lab
    # Overwrite the _get_observation() func in parent class just to save the pipes-on-screen info for later
    def _get_observation_features(self) -> np.ndarray:
        pipes = []
        for up_pipe, low_pipe in zip(self._upper_pipes, self._lower_pipes):
            # the pipe is behind the screen?
            if low_pipe["x"] > self._screen_width:
                pipes.append((self._screen_width, 0, self._screen_height))
            else:
                pipes.append(
                    (low_pipe["x"], (up_pipe["y"] + PIPE_HEIGHT), low_pipe["y"])
                )

        pipes = sorted(pipes, key=lambda x: x[0])
        self.raw_pipes = pipes  # store these pipes for later use in get_custom_reward()
        pos_y = self._player_y
        vel_y = self._player_vel_y
        rot = self._player_rot

        if self._normalize_obs:
            pipes = [
                (
                    h / self._screen_width,
                    v1 / self._screen_height,
                    v2 / self._screen_height,
                )
                for h, v1, v2 in pipes
            ]
            pos_y /= self._screen_height
            vel_y /= PLAYER_MAX_VEL_Y
            rot /= 90

        return (
            np.array(
                [
                    pipes[0][0],  # the last pipe's horizontal position
                    pipes[0][1],  # the last top pipe's vertical position
                    pipes[0][2],  # the last bottom pipe's vertical position
                    pipes[1][0],  # the next pipe's horizontal position
                    pipes[1][1],  # the next top pipe's vertical position
                    pipes[1][2],  # the next bottom pipe's vertical position
                    pipes[2][0],  # the next next pipe's horizontal position
                    pipes[2][1],  # the next next top pipe's vertical position
                    pipes[2][2],  # the next next bottom pipe's vertical position
                    pos_y,  # player's vertical position
                    vel_y,  # player's vertical velocity
                    rot,  # player's rotation
                ]
            ),
            None,
        )
    
    def get_custom_reward(self):
        """
        Return negative reward linearly based on how close pos_y of bird is to next pipe 
        on screen's gap.
        Manually sorts raw pipes because obs are pipes 1,2,3 and not sorted by next pipe,
        and may also be normalized.
        """
        pos_x = self._player_x
        pipe_clear = pos_x+PIPE_WIDTH
        # Find the next pipe that is ahead of the player
        next_pipe = [x for x in self.raw_pipes if x[0] > pipe_clear][0]
        # if the next pipe is already on screen,
        if next_pipe[0] < self._screen_width:
            # find the midpoint of pipe opening
            next_target = self.raw_pipes[0][1] + (self._pipe_gap/2)
            pos_y = self._player_y
            # using screen height normalizes reward between -1 and 0
            alt_rew = abs(pos_y-next_target)/self._screen_height*self.custom_reward_weight
            return alt_rew
        else:
            return 0
