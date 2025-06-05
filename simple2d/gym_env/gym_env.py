import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .sim import Simple2DSim
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
        self.render_mode = config.get('render_mode', 'rgb_array')
        self.sim = Simple2DSim(config)
        self.action_space = spaces.MultiDiscrete([3, 2])  # [(0: turn left, 1: turn right, 2: no turn), (0: decrease speed, 1: increase speed)] 
        self.observation_space = spaces.Box(
            low=np.array([-10, -10, 0, 0]), # [x, y, heading, velocity]
            high=np.array([10, 10, 2 * np.pi, np.finfo(np.float32).max]), 
            dtype=np.float32
        )
        self.step_count = 0

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = self.sim.reset()
        obs = np.array([self.state['position'][0], self.state['position'][1], self.state['heading'], self.state['velocity']], dtype=np.float32)
        return obs, {}

    def step(self, action):
        self.step_count += 1
        turn = action[0]
        speed_change = action[1] 
        self.state = self.sim.step([turn, speed_change])

        position = self.state['position']
        heading = self.state['heading']
        velocity = self.state['velocity']
        terminated = bool(np.linalg.norm(position) < self.goal_distance or np.max(np.abs(position)) >= 10)  # Consider done if within 0.1 units of the origin or too far
        reward = -np.linalg.norm(position)  # Negative distance to the origin as reward
        truncated = self.step_count >= self.horizon 
        obs = np.array([position[0], position[1], heading, velocity], dtype=np.float32)
        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode != 'rgb_array':
            return None
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

        # Draw walls around the -10 to 10 boundary in x and y
        wall_color = 'red'
        wall_thickness = 0.1
        walls = [
            plt.Line2D([-10, -10], [-10, 10], color=wall_color, linewidth=wall_thickness),  # Left wall
            plt.Line2D([10, 10], [-10, 10], color=wall_color, linewidth=wall_thickness),    # Right wall
            plt.Line2D([-10, 10], [-10, -10], color=wall_color, linewidth=wall_thickness),  # Bottom wall
            plt.Line2D([-10, 10], [10, 10], color=wall_color, linewidth=wall_thickness)     # Top wall
        ]
        for wall in walls:
            ax.add_line(wall)


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

