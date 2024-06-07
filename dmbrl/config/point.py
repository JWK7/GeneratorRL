import dmbrl.modeling.Model as Model
import dmbrl.utils.DataFunctions as DataFunctions

import numpy as np
from dotmap import DotMap
import gym
import dmbrl.env

import torch
import threading
import cv2


class PointConfigModule:
    ENV_NAME = "point"
    TASK_HORIZON = 200
    iteration = 50
    NROLLOUTS_PER_ITER = 1
    PLAN_HOR = 25
    MODEL_IN, MODEL_OUT = 6, 4
    GP_NINDUCING_POINTS = 200
    def __init__(self):
        self.exp_cfg = DotMap()
        self._create_exp_cfg()

        self.tool_cfg = DotMap()
        self._create_tool_cfg()

        self.log_cfg = DotMap()
        self._create_log_cfg()

    def _create_exp_cfg(self):
        self.exp_cfg.env = gym.make(self.ENV_NAME, render_mode="rgb_array")
        self.exp_cfg.iteration = 500
        self.exp_cfg.alpha = 0.00001
        self.exp_cfg.beta = 0.00001
        self.exp_cfg.batch_size = 5
        self.exp_cfg.num_generators = 3
        self.exp_cfg.image_dim_base = 8
        self.exp_cfg.image_dim_dim = 3
        self.exp_cfg.image_dim = (pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),)
        self.exp_cfg.generator_dim = (pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim))
        self.exp_cfg.step = 5

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
        self.log_cfg.I0 = torch.zeros((self.exp_cfg.batch_size,)+(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))
        self.log_cfg.Ix = torch.zeros((self.exp_cfg.batch_size,)+(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))
        self.log_cfg.deltaI = torch.zeros((self.exp_cfg.batch_size,)+(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))
        self.log_cfg.x = torch.zeros((self.exp_cfg.num_generators,self.exp_cfg.batch_size,1,1))
        pass

    def sample(self):
        print(torch.sum(self.log_cfg.I0))
        print(torch.sum(self.log_cfg.Ix))
        self.threadSampling()
        print(torch.sum(self.log_cfg.I0))
        print(torch.sum(self.log_cfg.Ix))
        print("hi")
        exit()
        return self._create_log_cfg.I0, self._create_log_cfg.Ix, self._create_log_cfg.Ix-self._create_log_cfg.I0, self._create_log_cfg.x
        # return torch.Tensor(I0),torch.Tensor(Ix),torch.Tensor(Ix-I0),torch.Tensor(x)
    
    def threadSampling(self):
        threads = []
        # for i in range(self.exp_cfg.iteration):
        for i in range(2):
            t = threading.Thread(target=self.threadSample,args=(i,))
            t.start()
            threads.append(t)
        for thread in threads:
            thread.join()
        print("fin")

    def threadSample(self,i):
        env = gym.make(self.ENV_NAME, render_mode="rgb_array")
        print(env.reset())
        self.log_cfg.I0[i] = self.renderImage(env)

        action = torch.randint(0,2,self.exp_cfg.num_generators)
        action[action==0] = -1
        self.log_cfg.x[:,i] = action

        for _ in range(self.exp_cfg.step): env.step(action)
        self.log_cfg.Ix[i] = self.renderImage(env)

    def renderImage(self,env):
        img = cv2.cvtColor(env.render(), cv2.COLOR_BGR2GRAY)
        return cv2.resize(img,(pow(self.exp_cfg.image_dim_base,self.exp_cfg.image_dim_dim),1))

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
            xG = x[0] * generators[0].GNet(I0.squeeze())
        else:
            # xG = generators[0].xNet(I0).detach() * generators[0].GNet(I0).detach()
            xG = x[0] * generators[0].GNet(I0.squeeze()).detach()

        if len(generators) == 1:
            return xG

        return xG + self.CombineGenerators(generator_target_index-1,generators[1:],I0,x[1:])
CONFIG_MODULE = PointConfigModule
