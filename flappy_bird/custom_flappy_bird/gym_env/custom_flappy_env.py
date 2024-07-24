"""Custom gymnasium env that inherits from FlappyBirdEnv"""

from typing import Dict, Tuple
import numpy as np
import gymnasium as gym
from numpy import ndarray
from flappy_bird_gymnasium import FlappyBirdEnv
from flappy_bird_gymnasium.envs.flappy_bird_env import FlappyBirdLogic

class CustomFlappyBirdEnv(FlappyBirdEnv):
    """
    Inherits a FlappyBirdEnv 0.3.0 for customization
    Uses the 0.3.0 version on pip for https://github.com/markub3327/flappy-bird-gymnasium/tree/v0.3.0
    """

    def __init__(self, env_config = {}, screen_size: Tuple[int, int] = (288, 512), audio_on: bool = False, 
                 normalize_obs: bool = False, pipe_gap: int = 100, bird_color: str = "yellow", pipe_color: str = "green", 
                 render_mode: str | None = None, background: str | None = "day"
                 ) -> None:
        # This enables env_configs passed through ray to overwrite  __init__ args
        screen_size = env_config.get('screen_size', screen_size)
        audio_on = env_config.get('audio_on', audio_on)
        normalize_obs = env_config.get('normalize_obs', normalize_obs)
        pipe_gap = env_config.get('pipe_gap', pipe_gap)
        bird_color = env_config.get('bird_color', bird_color)
        pipe_color = env_config.get('pipe_color', pipe_color)
        render_mode = env_config.get('render_mode', render_mode)
        background = env_config.get('background', background)
        super().__init__(screen_size, audio_on, normalize_obs, pipe_gap, bird_color, pipe_color, render_mode, background)
    
    def step(self, action: FlappyBirdLogic.Actions | int) -> Tuple[ndarray, float, bool, Dict]:
        obs, reward, terminal, _, info = super().step(action)

        # # TODO: if the episode is over, set color of the bird to 'red'
        # HINT: look for a self method that is obviously named in the parent env class. 


        return (
            obs,
            reward,
            terminal,
            False,
            info,
        )
    
    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed, options)

        # TODO: set the color of the bird back to "None"


        return obs, info
