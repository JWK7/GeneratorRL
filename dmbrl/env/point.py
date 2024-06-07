import math
from typing import Optional, Union

import numpy as np

import gym
from gym import utils
from gym.envs.mujoco import MujocoEnv
from gym.spaces import Box
import os


class PointEnv(MujocoEnv, utils.EzPickle):

    metadata = {
        "render_modes": [
            "human",
            "rgb_array",
            "depth_array",
        ],
        "render_fps": 7,
    }

    def __init__(
        self, **kwargs
    ):
        self._reset_noise_scale = 0.01
        observation_space = Box(low=-np.inf, high=np.inf, shape=(17,), dtype=np.float64)
        utils.EzPickle.__init__(self, **kwargs)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        MujocoEnv.__init__(self, '%s/assets/point.xml' % dir_path, 5, observation_space=observation_space, **kwargs)



    def _get_obs(self):
        position = self.data.qpos.flat.copy()
        velocity = self.data.qvel.flat.copy()

        # com_inertia = self.data.cinert.flat.copy()
        # com_velocity = self.data.cvel.flat.copy()

        # actuator_forces = self.data.qfrc_actuator.flat.copy()
        # external_contact_forces = self.data.cfrc_ext.flat.copy()

        return np.concatenate(
            (
                position,
                velocity,
                # com_inertia,
                # com_velocity,
                # actuator_forces,
                # external_contact_forces,
            )
        )

    def step(self, action):
        self.do_simulation(action, self.frame_skip)
        observation = self._get_obs()
        # reward = rewards - ctrl_cost
        reward =0
        terminated = 0
        info = {
        }

        if self.render_mode == "human":
            self.render()
        return observation, reward, terminated, False, info

    def reset_model(self):
        noise_low = -self._reset_noise_scale
        noise_high = self._reset_noise_scale

        qpos = self.init_qpos + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nq
        )
        qvel = self.init_qvel + self.np_random.uniform(
            low=noise_low, high=noise_high, size=self.model.nv
        )
        self.set_state(qpos, qvel)

        observation = self._get_obs()
        return observation
     

