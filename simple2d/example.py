"""Example script to test the Simple2D environment and render it using moviepy.
and also pygame for interactive rendering."""

import moviepy as mpy
from gym_env.gym_env import Simple2DEnv
from time import sleep

env = Simple2DEnv(render_mode="rgb_array")
obs = env.reset()
terminated = truncated = False
img_list = []
while not (terminated or truncated):
    action = env.action_space.sample()  # Random action for testing
    obs, reward, terminated, truncated, info = env.step(action)
    img_list.append(env.render())
    print(
        f"Obs: {obs}, Reward: {reward},\
        Terminated: {terminated}, Truncated: {truncated}"
    )
clip = mpy.ImageSequenceClip(img_list, fps=30)
clip.write_videofile("example.mp4", logger=None, audio=False)

env = Simple2DEnv(render_mode="human")
obs = env.reset()
terminated = truncated = False
while not (terminated or truncated):
    action = env.action_space.sample()  # Random action for testing
    obs, reward, terminated, truncated, info = env.step(action)
    env.render()
    sleep(0.01)  # Slow down rendering for visibility
    print(
        f"Obs: {obs}, Reward: {reward},\
        Terminated: {terminated}, Truncated: {truncated}"
    )
env.close()
