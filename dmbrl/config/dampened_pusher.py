import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import numpy as np
from dotmap import DotMap

# import torch
# import threading
# import cv2


class DampenedHumanoidModule:
    def __init__(self):
        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env_name = "dampened_pusher"
        self.exp_cfg.iteration = 500
        self.exp_cfg.alpha_x = 0.01
        self.exp_cfg.alpha_G = 0.01
        self.exp_cfg.batch_size = 2
        self.exp_cfg.step = 5

        self.exp_cfg.full_state_dim = 14
        self.exp_cfg.qpos_state_dim = 7
        self.exp_cfg.num_generators = self.exp_cfg.qpos_state_dim
        self.exp_cfg.generator_dim = (2,2)

    def _create_tool_cfg(self):
        for i in range(self.exp_cfg.num_generators):
            if i == 0:
                self.tool_cfg.nn = Model(self.exp_cfg.alpha_G,self.exp_cfg.alpha_x,self.exp_cfg.qpos_state_dim,self.exp_cfg.generator_dim),
            else:
                self.tool_cfg.nn += Model(self.exp_cfg.alpha_G,self.exp_cfg.alpha_x,self.exp_cfg.qpos_state_dim, self.exp_cfg.generator_dim),

        self.tool_cfg.sample = self.__sample

    def _create_log_cfg(self):
        pass

    def __sample(self):
        return DataFunctions.gym_sample(self.exp_cfg)

CONFIG_MODULE = DampenedHumanoidModule