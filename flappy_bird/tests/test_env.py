import pytest
import gymnasium
from gymnasium.utils.env_checker import check_env
from gymnasium.envs.registration import register

register(
    id="customflappybird",
    entry_point="custom_flappy_bird.gym_env.custom_flappy_env:CustomFlappyBirdEnv",
)
@pytest.fixture
def test_env():
    return gymnasium.make("customflappybird")

def test_valid_env(test_env):
    check_env(test_env)
