"""
Script to demo two episodes of the parent gym env
FYI this file cannot see modules in ../ for importing when run as a 'script' (press play button in vscode).
Pylint will say you can import them, but you would have to run this as a 'module', if you were to try importing 
above here.

I.e. if you add:
import custom_flappy_bird

This works:
    python -m custom_flappy_bird.scripts.demo
This does NOT work:
    python ./custom_flappy_bird/scripts/demo.py 
"""

import numpy as np
import flappy_bird_gymnasium
import gymnasium
import imageio
# import custom_flappy_bird


env = gymnasium.make("FlappyBird-v0", render_mode="rgb_array") # pip library env

obs, _ = env.reset()
images = []
while True:
    # Random action from action space:
    action = env.action_space.sample()
    obs, reward, terminated, _, info = env.step(action)
    # if terminated: env.set_color('red')
    images.append(env.render())
    
    # Checking if the player is still alive
    if terminated:
        print('terminated')
        break

imageio.mimsave('./flappy1.gif', images)

obs, _ = env.reset()
# env.set_color(None)
images = []
while True:
    # Flap just 5% of the time
    action = np.random.choice([0,1], p=[0.95, 0.05])
    # Processing:
    obs, reward, terminated, _, info = env.step(action)
    # if terminated: env.set_color('red')
    images.append(env.render())
    
    # Checking if the player is still alive
    if terminated:
        print('terminated')
        break

imageio.mimsave('./flappy2.gif', images)


env.close()

