import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import pandas as pd
import numpy as np
from dotmap import DotMap

import torch
import torch.nn as nn

class Image1DConfigModule:

    def __init__(self):

        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env = 'Image1D'
        self.exp_cfg.iteration = 1000
        self.exp_cfg.alpha = 0.0001
        self.exp_cfg.beta = 0.0001
        self.exp_cfg.data_size = 50
        self.exp_cfg.num_generators = 2
        self.exp_cfg.image_dim = (20*20,)
        self.exp_cfg.generator_dim = (20*20,20*20)
        self.exp_cfg.T = 0.5
        self.exp_cfg.ground_truth_generator = np.reshape(pd.read_csv("dmbrl/assets/Translation1D20Pixels.csv",header=None).to_numpy(),(20,20))
    
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

    def sample(self):
        I0 = DataFunctions.NoiseImage(((self.exp_cfg.data_size,)+(20,20)))
        # x , Ix = DataFunctions.Translation1DImage(I0,self.exp_cfg.T,self.exp_cfg.ground_truth_generator)

        x,Ix = DataFunctions.Translation2DImage(I0,self.exp_cfg.T,self.exp_cfg.ground_truth_generator)
        I0= I0.reshape((self.exp_cfg.data_size,)+(20*20,1))
        Ix = Ix.reshape((self.exp_cfg.data_size,)+(20*20,1))


        return torch.Tensor(I0),torch.Tensor(Ix),torch.Tensor(Ix-I0),torch.Tensor(x)

    def UpdateG(self,generator_target_index,generators,I0,deltaI,x):
        xG = self.CombineGenerators(generator_target_index,generators,I0,x)
        xGI0 = torch.linalg.matmul(xG.float(),I0)

        loss = nn.MSELoss()
        generators[generator_target_index].optimG.zero_grad()
        grad = loss(xGI0.squeeze(),deltaI.squeeze())
        grad.backward()
        generators[generator_target_index].optimG.step()

        return grad.detach()

    def UpdateX(self,generator_target_index,generators,I0,deltaI,x):
        pass

    def CombineGenerators(self,generator_target_index,generators,I0,x):
        #Keeps gradience for targeted generator
        if generator_target_index == 0:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0)
            xG = x * generators[0].GNet(I0.squeeze())
        else:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0).detach()
            xG = x * generators[0].GNet(I0.squeeze()).detach()

        if len(generators) == 1:
            return xG

        return xG + self.CombineGenerators(generator_target_index-1,generators[1:],I0,x)


CONFIG_MODULE = Image1DConfigModule