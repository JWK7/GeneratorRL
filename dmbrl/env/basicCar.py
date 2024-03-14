"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""
import math
from typing import Optional, Union

import numpy as np

import gym
from gym import logger, spaces
from gym.envs.classic_control import utils
from gym.error import DependencyNotInstalled


class basicCarEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):

        self.x_threshold = 300
        self.y_threshold = 200
        self.velocity_threshold = 10
        self.torque_threshold = 2
        self.angularchange_threshold = 0.1


        #forward/backward
        #angular change
        high_action = np.array(
            [
                self.torque_threshold,
                self.angularchange_threshold,
            ],
            dtype=np.float32,
        )

        #car location (x,y)
        #speed max
        #angle
        high_observation = np.array(
            [
                self.x_threshold,
                self.y_threshold,
                self.velocity_threshold,
                math.pi,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-high_action, high=high_action, shape=(2,), dtype=np.float32
        )
        self.observation_space = spaces.Box(-high_observation, high_observation, dtype=np.float32)

        self.render_mode = render_mode

        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None

        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(
            action
        ), f"{action!r} ({type(action)}) invalid"

        x_position, y_position, velocity, angle = self.state

        velocity += action[0]
        angle += action[1]
        x_position += math.cos(angle) * velocity
        y_position += math.sin(angle) * velocity
        self.state = (x_position,y_position,velocity,angle)

        return np.array(self.state,dtype=np.float32), False, False, False, {}

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        super().reset(seed=seed)

        high = np.array(
            [
                self.x_threshold//2,
                self.y_threshold//2,
                0,
                0,
            ]
        )
        self.state = self.np_random.uniform(low=-high, high=high, size=(4,))
        self.steps_beyond_terminated = None

        if self.render_mode == "human":
            self.render()
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            gym.logger.warn(
                "You are calling render method without specifying any render mode. "
                "You can specify the render_mode at initialization, "
                f'e.g. gym("{self.spec.id}", render_mode="rgb_array")'
            )
            return

        try:
            import pygame
            from pygame import gfxdraw
        except ImportError:
            raise DependencyNotInstalled(
                "pygame is not installed, run `pip install gym[classic_control]`"
            )

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode(
                    (self.screen_width, self.screen_height)
                )
            else:  # mode == "rgb_array"
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold *2
        scale = self.screen_width / world_width
        carwidth = 40
        carheight = 20

        self.surf = pygame.Surface((self.screen_width, self.screen_height))
        self.surf.fill((255, 255, 255))

        x_pos = self.state[0]
        y_pos = self.state[1]
        rad = self.state[3]

        l, r, t, b = -carwidth / 2, carwidth / 2, carheight, 0
        coords = []
        for c in [(l, b), (l, t), (r, t), (r, b)]:
            c = pygame.math.Vector2(c).rotate_rad(rad)
            coords.append(
                (
                    c[0] + (x_pos + self.x_threshold) * scale,
                    c[1] + (y_pos + self.y_threshold) * scale,
                )
            )

        gfxdraw.aapolygon(self.surf, coords, (0, 0, 0))
        gfxdraw.filled_polygon(self.surf, coords, (0, 0, 0))



        self.surf = pygame.transform.flip(self.surf, False, True)
        self.screen.blit(self.surf, (0, 0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()

        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.screen)), axes=(1, 0, 2)
            )
            
    def close(self):
        if self.screen is not None:
            import pygame

            pygame.display.quit()
            pygame.quit()
            self.isopen = False
