import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import numpy as np
from dotmap import DotMap


import torch
import torch.nn as nn
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
        self.exp_cfg.iteration = 100
        self.exp_cfg.alpha_x = 0.1
        self.exp_cfg.alpha_G = 0.1
        self.exp_cfg.batch_size = 2
        self.exp_cfg.step = 100

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
        self.tool_cfg.UpdateGradienceG = self.UpdateG

    def _create_log_cfg(self):
        self.log_cfg.xloss = []
        self.log_cfg.Gloss = []

        pass

    def __sample(self):
        return DataFunctions.gym_sample(self.exp_cfg)

    def UpdateGradienceG(self,data):
        G = self.tool_cfg.nn[0].GNet((data[2].unsqueeze(-1)).to(self.tool_cfg.nn[0].device))
        # x = net.xNet((data[2].unsqueeze(-1)).to(net.device)).detach()
        # xG = x * G
        # xG = G
        # I0 = torch.zeros((data.shape[1],2,2))
        # I0[:,0,1] = -data[0]
        # I0[:,1,0] = data[0]
        # I0 = I0.to(net.device)

        deltaI = torch.ones((data.shape[1],2,2))
        # deltaI[:,0,1] = -data[2]
        # deltaI[:,1,0] = data[2]
        # deltaI = deltaI.to(net.device)
        # xGI0 = torch.matmul(xG,I0)

        # print(G)
        # print(deltaI)
        # # exit()

        loss = nn.MSELoss()
        self.tool_cfg.nn[0].optimG.zero_grad()
        grad = loss(G.squeeze(),(deltaI).squeeze().to(self.tool_cfg.nn[0].device))
        grad.backward()
        self.tool_cfg.nn[0].optimG.step()
        print(grad)

        return grad.detach().to('cpu')



    def UpdateG(self,data):
        for _ in range(100):
            self.UpdateGradienceG(data[:,:,0])
        # for i,net in enumerate(params.tool_cfg.nn):
        #     if i < 1:
        #         params.log_cfg.Gloss.append(UpdateGradienceG(net,data[:,:,i]))


CONFIG_MODULE = DampenedHumanoidModule