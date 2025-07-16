"""Simple 2d pos, vel, heading dynamics with no friction etc"""

import numpy as np


class Simple2DSim:
    def __init__(self, config={}):
        self.position = np.zeros(2)  # [x, y]
        self.velocity = 0.0  # scalar velocity
        self.heading = 0.0  # angle in radians
        self.dt = config.get("dt", 1.0)
        self.dV = config.get("dV", 0.1)
        self.reset()

    def reset(self):
        while True:
            self.position = np.random.uniform(-10, 10, size=2)
            if np.linalg.norm(self.position) >= 5:
                break
        self.velocity = 0.0
        # self.velocity = np.random.uniform(0, 1)
        self.heading = np.random.uniform(0, 2 * np.pi)
        return self._get_state()

    def step(self, action):
        turn, speed_change = action

        # Update heading
        if turn == 0:  # turn left
            self.heading -= np.pi / 18  # turn by 10 degrees
        elif turn == 1:  # turn right
            self.heading += np.pi / 18  # turn by 10 degrees
        elif turn == 2:  # no turn
            pass
        else:
            raise ValueError("Invalid turn action")

        if self.heading < 0:
            self.heading += 2 * np.pi
        elif self.heading >= 2 * np.pi:
            self.heading -= 2 * np.pi

        # Update velocity
        if speed_change == 0:  # slow down
            self.velocity = max(0, self.velocity - self.dV)
        elif speed_change == 1:  # speed up
            self.velocity += self.dV
        elif speed_change == 2:  # no change
            pass
        else:
            raise ValueError("Invalid speed change action")

        # Update position
        self.position[0] += self.velocity * np.cos(self.heading) * self.dt
        self.position[1] += self.velocity * np.sin(self.heading) * self.dt
        self.position = np.clip(
            self.position, -10, 10
        )  # Clip position to be within [-10, 10] walls

        return self._get_state()

    def _get_state(self):
        return {
            "position": self.position.copy(),
            "velocity": self.velocity,
            "heading": self.heading,
        }
