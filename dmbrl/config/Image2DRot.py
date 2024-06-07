import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import pandas as pd
import numpy as np
from dotmap import DotMap

import torch
import torch.nn as nn

class Image2DRotConfigModule:

    def __init__(self):

        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env = 'Image2DRot'
        self.exp_cfg.iteration = 15000
        self.exp_cfg.alpha = 0.00005
        self.exp_cfg.beta = 0.00001
        self.exp_cfg.data_size = 64
        self.exp_cfg.num_generators = 1
        self.exp_cfg.image_dim = (30*30,)
        self.exp_cfg.generator_dim = (30*30,30*30)
        self.exp_cfg.T = 30
    
    def _create_tool_cfg(self):
        for i in range(self.exp_cfg.num_generators):
            if i == 0:
                self.tool_cfg.nn = Model.Model1D(self.exp_cfg.alpha,self.exp_cfg.beta,self.exp_cfg.image_dim,self.exp_cfg.generator_dim),
            else:
                self.tool_cfg.nn += Model.Model1D(self.exp_cfg.alpha,self.exp_cfg.beta,self.exp_cfg.image_dim,self.exp_cfg.generator_dim),

        self.tool_cfg.sample = self.sample
        self.tool_cfg.UpdateG = self.UpdateG
        self.tool_cfg.UpdateX = self.UpdateX
        self.tool_cfg.CombineGenerators = self.CombineGenerators
    
    def _create_log_cfg(self):
        pass

    def sample(self,x=1):
        I0 = DataFunctions.NoiseImage(((self.exp_cfg.data_size,)+(20,20)))

        x,Ix = DataFunctions.Rotate2D(I0,self.exp_cfg.T*x)
        I0= I0.reshape((self.exp_cfg.data_size,)+(20*20,1))
        Ix = Ix.reshape((self.exp_cfg.data_size,)+(20*20,1))

        return torch.Tensor(I0),torch.Tensor(Ix),torch.Tensor(Ix-I0),torch.Tensor(x)

    def UpdateG(self,generator_target_index,generators,I0,deltaI,x):
        xG = self.CombineGenerators(generator_target_index,generators,deltaI,x)

        # G = generators[0].GNet(deltaI.squeeze().to('cuda'))
        # G = torch.roll(G,1,0)
        # xG = x[0].to("cuda")*G

        expxG = DataFunctions.exp_series(7,xG)

        # xGI0 = torch.linalg.matmul(xG.float(),I0.to('cuda'))
        xGI0=  torch.linalg.matmul(expxG.float(),I0.to(generators[0].device))

        loss = nn.MSELoss()
        generators[generator_target_index].optimG.zero_grad()
        grad = loss(xGI0.squeeze(),(deltaI+I0).squeeze().to(generators[0].device))
        grad.backward()
        generators[generator_target_index].optimG.step()

        return grad.detach()

    def UpdateX(self,generator_target_index,generators,I0,deltaI,x):
        pass

    def CombineGenerators(self,generator_target_index,generators,deltaI,x):
        #Keeps gradience for targeted generator
        if generator_target_index == 0:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0)
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device))
        else:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0).detach()
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device)).detach()
        if len(generators) == 1:
            return xG

        return xG + self.CombineGenerators(generator_target_index-1,generators[1:],I0,x[1:])


CONFIG_MODULE = Image2DRotConfigModule