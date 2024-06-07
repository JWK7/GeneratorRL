import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import pandas as pd
import numpy as np
from dotmap import DotMap

import torch
import torch.nn as nn

class Image3DRotConfigModule:

    def __init__(self):

        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env = 'Image3DRot'
        self.exp_cfg.iteration = 30000
        self.exp_cfg.alpha = 0.00001
        self.exp_cfg.beta = 0.00001
        self.exp_cfg.data_size = 32
        self.exp_cfg.num_generators = 3
        self.exp_cfg.image_dim_base = 8
        self.exp_cfg.image_dim_dim = 3
        self.exp_cfg.image_dim = (pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),)
        self.exp_cfg.generator_dim = (pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim))
        self.exp_cfg.T = 5
        # self.exp_cfg.ground_truth_generator = np.reshape(pd.read_csv("dmbrl/assets/Translation1D20Pixels.csv",header=None).to_numpy(),(20,20))
    
    def _create_tool_cfg(self):

        for i in range(self.exp_cfg.num_generators):
            if i == 0:
                self.tool_cfg.nn = Model.Model1D(self.exp_cfg.alpha,self.exp_cfg.beta,self.exp_cfg.image_dim,self.exp_cfg.generator_dim,10000,i),
            else:
                self.tool_cfg.nn += Model.Model1D(self.exp_cfg.alpha,self.exp_cfg.beta,self.exp_cfg.image_dim,self.exp_cfg.generator_dim,10000,i),

        self.tool_cfg.sample = self.sample
        self.tool_cfg.UpdateG = self.UpdateG
        self.tool_cfg.UpdateX = self.UpdateX
        self.tool_cfg.CombineGenerators = self.CombineGenerators
    
    def _create_log_cfg(self):
        pass

    def sample(self):
        I0 = DataFunctions.NoiseImage(((self.exp_cfg.data_size,)+(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_base)))
        x,Ix = DataFunctions.Rotate3D(I0,self.exp_cfg.T)
        I0= I0.reshape((self.exp_cfg.data_size,)+(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))
        Ix = Ix.reshape((self.exp_cfg.data_size,)+(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))

        # I0 /= 255
        # Ix /= 255
        return torch.Tensor(I0),torch.Tensor(Ix),torch.Tensor(Ix-I0),torch.Tensor(x)

    def UpdateG(self,generator_target_index,generators,I0,deltaI,x):
        jacobian , xGI0 = self.CombineGenerators_(generator_target_index,generators,I0,deltaI,x)
    
        # xGI0 = torch.linalg.matmul(xG.float(),I0)

        loss = nn.MSELoss()
        generators[generator_target_index].optimG.zero_grad()
        # print(torch.mean(torch.abs(jacobian)).to(generators[generator_target_index].device))
        
        grad = loss(xGI0.squeeze(),deltaI.squeeze().to(generators[generator_target_index].device))#+torch.sum(torch.abs(jacobian)).to(generators[generator_target_index].device)
        # print(grad)
        grad.backward()
        generators[generator_target_index].optimG.step()

        # return grad.detach()
        return (grad.detach(),torch.sum(torch.abs(jacobian)))

    def CombineGenerators_(self,generator_target_index,generators,I0,deltaI,x):

        Gs = self.GetGeneratorPredictions(generator_target_index,generators,deltaI,x)
        # print(torch.sum(torch.abs(Gs[0]-Gs[1])))

        # for i in Gs: i.to(generators[generator_target_index].device)
        

        # xGI0 =torch.linalg.matmul(Gs[0]+ Gs[1] + Gs[2],I0.to(generators[generator_target_index].device))
        xGI0 =torch.linalg.matmul(Gs[generator_target_index],I0.to(generators[generator_target_index].device))
        # xGI0 = I0.to('cuda')
        # for i in Gs:
            # xGI0 = torch.linalg.matmul(i.float(),xGI0)
        # return jacobian , xGI0
        # print(xGI0)
        jacobian = self.GetJacobian(Gs,generator_target_index)

        return jacobian , xGI0
    
    def GetJacobian(self,Gs,generator_target_index):
        return torch.matmul(Gs[0].detach(),torch.matmul(Gs[1].detach(),Gs[2].detach()))+ torch.matmul(Gs[1].detach(),torch.matmul(Gs[2].detach(),Gs[0].detach()))+torch.matmul(Gs[2].detach(),torch.matmul(Gs[0].detach(),Gs[1].detach()))


    def UpdateX(self,generator_target_index,generators,I0,deltaI,x):
        pass

    def GetGeneratorPredictions(self,generator_target_index,generators,deltaI,x):
        #Keeps gradience for targeted generator
        if generator_target_index == 0:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0)
            # xG = x[0] * generators[0].GNet(deltaI.squeeze())
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device))
        else:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0).detach()
            # xG = x[0] * generators[0].GNet(deltaI.squeeze()).detach()
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device)).detach()

        if len(generators) == 1:
            return [xG]

        return [xG] + self.GetGeneratorPredictions(generator_target_index-1,generators[1:],deltaI,x[1:])
    
    def CombineGenerators(self,generator_target_index,generators,deltaI,x):
        #Keeps gradience for targeted generator
        if generator_target_index == 0:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0)
            # xG = x[0] * generators[0].GNet(deltaI.squeeze())
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device))
        else:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0).detach()
            # xG = x[0] * generators[0].GNet(deltaI.squeeze()).detach()
            xG = x[0].to(generators[0].device) * generators[0].GNet(deltaI.squeeze().to(generators[0].device)).detach()

        if len(generators) == 1:
            return xG

        return xG + self.CombineGenerators(generator_target_index-1,generators[1:],deltaI,x[1:])


CONFIG_MODULE = Image3DRotConfigModule