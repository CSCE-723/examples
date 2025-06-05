import moviepy as mpy
from gym_env.gym_env import Simple2DEnv

env = Simple2DEnv()
obs = env.reset()
terminated = truncated = False
img_list = []
while not (terminated or truncated):
    action = env.action_space.sample()  # Random action for testing
    obs, reward, terminated, truncated, info = env.step(action)
    img_list.append(env.render())
    print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
clip = mpy.ImageSequenceClip(img_list, fps=30)
clip.write_videofile('example.mp4', logger=None, audio=False)
