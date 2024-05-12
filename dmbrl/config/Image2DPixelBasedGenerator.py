import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import pandas as pd
import numpy as np
from dotmap import DotMap

import torch
import torch.nn as nn

class Image2DXYConfigModule:

    def __init__(self):

        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env = 'Image2DXY'
        self.exp_cfg.iteration = 100
        self.exp_cfg.alpha = 0.0001
        self.exp_cfg.beta = 0.0001
        self.exp_cfg.data_size = 50
        self.exp_cfg.num_generators = 2
        self.exp_cfg.image_dim = (20*20,)
        self.exp_cfg.generator_dim = (3,3)
        self.exp_cfg.T = 1
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
        xGI0 = self.ApplyGenerator(xG.float(),I0)

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
            xG = x[0] * generators[0].GNet(I0.squeeze())

        else:
            xG = x[0] * generators[0].GNet(I0.squeeze()).detach()
        if len(generators) == 1:
            return xG

        return xG + self.CombineGenerators(generator_target_index-1,generators[1:],I0,x[1:])

    def ApplyGenerator(self,generator,I0):
        xGI0 = torch.zeros(I0.shape, dtype=generator.dtype)
        xGI0[:,0,0] = generator[:,0,0]
        xGI0[:,0,0] = 0
        for i in range((I0.shape[1])):
            for j in range((I0.shape[2])):
                loc = torch.matmul(generator,torch.tensor(([[[j],[i],[1]]])).float())[:,:,0].int()
                xGI0[:,loc[:,1]%I0.shape[1],loc[:,0]%I0.shape[2]] = I0[:,i,j]

        # print(generator.data)
        # xGI0.grad_fn.copy_(generator.grad_fn)
        # print(xGI0.grad_fn)
        # xGI0.grad.copy_(generator.grad_fn)
        return xGI0


CONFIG_MODULE = Image2DXYConfigModule