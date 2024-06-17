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
        self.exp_cfg.env_name = "dampened_humanoid"
        self.exp_cfg.iteration = 500
        self.exp_cfg.alpha_x = 0.01
        self.exp_cfg.alpha_G = 0.01
        self.exp_cfg.batch_size = 5
        self.exp_cfg.step = 5

        self.exp_cfg.state_dim = (17,)
        self.exp_cfg.num_generators = self.exp_cfg.state_dim[0]
        self.exp_cfg.generator_dim = (3,3)

    def _create_tool_cfg(self):
        for i in range(self.exp_cfg.num_generators):
            if i == 0:
                self.tool_cfg.nn = Model(self.exp_cfg.alpha_G,self.exp_cfg.alpha_x,self.exp_cfg.state_dim,self.exp_cfg.generator_dim),
            else:
                self.tool_cfg.nn += Model(self.exp_cfg.alpha_G,self.exp_cfg.alpha_x,self.exp_cfg.state_dim, self.exp_cfg.generator_dim),

        self.tool_cfg.sample = self.sample
        # self.tool_cfg.UpdateG = self.UpdateG
        # self.tool_cfg.UpdateX = self.UpdateX
        # self.tool_cfg.CombineGenerators = self.CombineGenerators

    def _create_log_cfg(self):
        pass

    def sample(self):
        DataFunctions.gym_sample(self.exp_cfg)

CONFIG_MODULE = DampenedHumanoidModule