import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .sim import Simple2DSim
import pygame
import math


class Simple2DEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(
        self,
        render_mode="rgb_array",
        config={},
        horizon=500,
        goal_distance=1.0,
        max_speed=1.0,
    ):
        super().__init__()
        self.horizon = config.get("horizon", horizon)
        self.goal_distance = config.get("goal_distance", goal_distance)
        self.max_speed = config.get("max_speed", max_speed)
        self.render_mode = config.get("render_mode", render_mode)
        self.sim = Simple2DSim(config)
        # [(0: turn left, 1: turn right, 2: no turn),
        # (0: decrease speed, 1: increase speed)]
        self.action_space = spaces.MultiDiscrete([3, 3])
        self.observation_space = spaces.Box(
            # [x, y, cos(heading), sin(heading), velocity]
            low=np.array([-10.0, -10.0, -1.0, -1.0, 0.0]),
            high=np.array([10.0, 10.0, 1.0, 1.0, max_speed]),
            dtype=np.float32,
        )
        self.step_count = 0

        # Pygame rendering setup
        self.window_size = 600  # Size of the pygame window
        # Scale factor (30 world units = window_size pixels)
        self.scale = self.window_size / 30
        self.window = None
        self.clock = None

    def reset(self, seed=None, options=None):
        self.step_count = 0
        self.state = self.sim.reset()
        obs = np.array(
            [
                self.state["position"][0],
                self.state["position"][1],
                np.cos(self.state["heading"]),
                np.sin(self.state["heading"]),
                self.state["velocity"],
            ],
            dtype=np.float32,
        )
        return obs, {}

    def step(self, action):
        self.step_count += 1
        turn = action[0]
        speed_change = action[1]
        self.state = self.sim.step([turn, speed_change])

        position = self.state["position"]
        heading = self.state["heading"]
        velocity = self.state["velocity"]
        hit_goal = np.linalg.norm(position) < self.goal_distance
        terminated = bool(
            hit_goal or np.max(np.abs(position)) >= 10
        )  # Consider done if within 0.1 units of the origin or too far
        truncated = self.step_count >= self.horizon
        obs = np.array(
            [position[0], position[1], np.cos(heading), np.sin(heading), velocity],
            dtype=np.float32,
        )
        reward = self._get_reward(position)
        return obs, reward, terminated, truncated, {}

    def _get_reward(self, position):
        """Calculate reward based on position."""
        dist = np.linalg.norm(position)
        # reward = np.exp(-dist)  # Exponential decay based on distance
        # reward = -dist  # Negative distance as reward
        var = 1.0
        reward = math.exp(-0.5 / var * dist**2)  # Gaussian reward based on distance
        if dist < self.goal_distance:
            reward += 100
        if np.max(np.abs(position)) >= 10:
            reward -= 100
        return reward

    def render(self):
        if self.render_mode not in ["human", "rgb_array"]:
            return None

        # Initialize pygame if not already done
        if self.window is None:
            pygame.init()
            if self.render_mode == "human":
                window_size = (self.window_size, self.window_size)
                self.window = pygame.display.set_mode(window_size)
                pygame.display.set_caption("Simple2D Environment")
            else:
                self.window = pygame.Surface((self.window_size, self.window_size))
            self.clock = pygame.time.Clock()

        # Fill background with white
        self.window.fill((255, 255, 255))

        # Helper function to convert world coordinates to screen coordinates
        def world_to_screen(world_pos):
            x, y = world_pos
            screen_x = int((x + 15) * self.scale)
            screen_y = int((15 - y) * self.scale)  # Flip Y axis
            return (screen_x, screen_y)

        # Draw boundary walls (red lines)
        wall_color = (255, 0, 0)  # Red
        wall_thickness = 3

        # Convert boundary coordinates to screen coordinates
        top_left = world_to_screen((-10, 10))
        top_right = world_to_screen((10, 10))
        bottom_left = world_to_screen((-10, -10))
        bottom_right = world_to_screen((10, -10))

        # Draw walls
        # Top wall
        pygame.draw.line(self.window, wall_color, top_left, top_right, wall_thickness)
        # Bottom wall
        pygame.draw.line(
            self.window, wall_color, bottom_left, bottom_right, wall_thickness
        )
        # Left wall
        pygame.draw.line(self.window, wall_color, top_left, bottom_left, wall_thickness)
        # Right wall
        pygame.draw.line(
            self.window, wall_color, top_right, bottom_right, wall_thickness
        )

        # Draw goal circle (green)
        goal_center = world_to_screen((0, 0))
        goal_radius = int(self.goal_distance * self.scale)
        goal_color = (0, 255, 0, 128)  # Green with transparency

        # Create a surface for the transparent circle
        circle_surface = pygame.Surface(
            (goal_radius * 2, goal_radius * 2), pygame.SRCALPHA
        )
        pygame.draw.circle(
            circle_surface, goal_color, (goal_radius, goal_radius), goal_radius
        )
        blit_pos = (goal_center[0] - goal_radius, goal_center[1] - goal_radius)
        self.window.blit(circle_surface, blit_pos)

        # Draw agent as a triangle (blue)
        position = self.state["position"]
        heading = self.state["heading"]
        agent_center = world_to_screen(position)
        agent_color = (0, 0, 255)  # Blue

        # Calculate triangle vertices
        triangle_size = int(0.5 * self.scale)

        # Triangle points relative to center (pointing up initially)
        relative_points = [
            (0, -triangle_size),  # Top point
            (-triangle_size * 0.8, triangle_size * 0.6),  # Bottom left
            (triangle_size * 0.8, triangle_size * 0.6),  # Bottom right
        ]

        # Rotate points based on heading
        rotated_points = []
        for px, py in relative_points:
            # Rotate by heading angle
            cos_h = math.cos(heading)
            sin_h = math.sin(heading)
            rotated_x = px * cos_h - py * sin_h
            rotated_y = px * sin_h + py * cos_h

            # Translate to agent position
            final_x = agent_center[0] + rotated_x
            final_y = agent_center[1] + rotated_y
            rotated_points.append((final_x, final_y))

        pygame.draw.polygon(self.window, agent_color, rotated_points)

        if self.render_mode == "human":
            pygame.display.flip()
            self.clock.tick(30)  # 30 FPS
            return None
        elif self.render_mode == "rgb_array":
            # Convert pygame surface to numpy array
            rgb_array = pygame.surfarray.array3d(self.window)
            rgb_array = np.transpose(rgb_array, (1, 0, 2))  # Swap x and y axes
            return rgb_array

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None
            self.clock = None
