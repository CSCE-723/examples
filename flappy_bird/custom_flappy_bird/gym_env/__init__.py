# """An optional way to register your environment.
# This will run when gym_env module is imported"""
# import os
# import sys

# # Add package to PYTHONPATH
# from custom_flappy_bird import ROOT_DIR
# sys.path.append(ROOT_DIR)
# os.chdir(ROOT_DIR)

# from ray.tune import register_env
# from gym_env.custom_flappy_env import CustomFlappyBirdEnv
# register_env("flappybird", CustomFlappyBirdEnv)
