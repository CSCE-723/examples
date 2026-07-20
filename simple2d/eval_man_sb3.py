"""Here is one episode using the trained algorithm to show a pygame episode"""
from time import sleep
from gym_env.gym_env import Simple2DEnv
from stable_baselines3 import PPO

episode_reward = 0
reward = 0
step = 0
env = Simple2DEnv(render_mode='human')  # get the environment from the model
obs, info = env.reset()
model = PPO.load("/workspace/sb3_logs/ppo_simple2d_final.zip")  # load the model
terminated = truncated = False
while not terminated and not truncated:
    # use the trained model to predict the action
    action = model.predict(obs, deterministic=True)[0]
    obs, reward, terminated, truncated, info = env.step(action)
    episode_reward += reward
    env.render()
    step += 1
    sleep(0.01)  # sleep to slow down the rendering a bit

print('episode reward was: ', episode_reward)
print('steps taken: ', step)
env.close()
