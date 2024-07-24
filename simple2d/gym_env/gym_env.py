import gymnasium as gym
from gymnasium import spaces
import numpy as np
from sim import Simple2DSim
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, RegularPolygon, Circle
import io
from PIL import Image

class Simple2DEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, config={}):
        super().__init__()
        self.horizon = config.get('horizon', 500)
        self.goal_distance = config.get('goal_distance', 1)
        self.sim = Simple2DSim(config)
        self.action_space = spaces.MultiDiscrete([3, 2])  # 0: turn left, 1: turn right, 2: no turn
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0]), 
            high=np.array([10, 10, 2 * np.pi]), 
            dtype=np.float32
        )
        self.step_count = 0

    def reset(self):
        self.step_count = 0
        self.state = self.sim.reset()
        return np.array([self.state['position'][0], self.state['position'][1], self.state['heading']], dtype=np.float32)

    def step(self, action):
        self.step_count += 1
        turn = action[0]
        speed_change = action[1] 
        self.state = self.sim.step([turn, speed_change])

        position = self.state['position']
        heading = self.state['heading']
        terminated = np.linalg.norm(position) < self.goal_distance or np.max(np.abs(position)) > 10  # Consider done if within 0.1 units of the origin or too far
        reward = -np.linalg.norm(position)  # Negative distance to the origin as reward
        truncated = self.step_count >= self.horizon 

        return np.array([position[0], position[1], heading], dtype=np.float32), reward, terminated, truncated, {}

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xlim(-15, 15)
        ax.set_ylim(-15, 15)
        ax.set_aspect('equal')

        # Create a triangle to represent the agent
        position = self.state['position']
        heading = self.state['heading']
        triangle = RegularPolygon(
            (position[0], position[1]), 
            numVertices=3, 
            radius=0.5, 
            orientation=heading - np.pi / 2,  # Adjust orientation to match heading
            color='blue'
        )
        goal_circle = Circle((0, 0), self.goal_distance, color='green', alpha=0.5)
        ax.add_patch(triangle)
        ax.add_patch(goal_circle)

        # Convert plot to RGB array
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        img = Image.open(buf)
        rgb_array = np.array(img)

        plt.close(fig)
        return rgb_array

    def close(self):
        pass

# Example
if __name__ == "__main__":
    import moviepy.editor as mpy
    env = Simple2DEnv()
    obs = env.reset()
    terminated = truncated = False
    img_list = []
    while not terminated or truncated:
        action = env.action_space.sample()  # Random action for testing
        obs, reward, terminated, truncated, info = env.step(action)
        img_list.append(env.render())
        print(f"Obs: {obs}, Reward: {reward}, Terminated: {terminated}, Truncated: {truncated}")
    clip = mpy.ImageSequenceClip(img_list, fps=30)
    clip.write_videofile('example.mp4', logger=None)