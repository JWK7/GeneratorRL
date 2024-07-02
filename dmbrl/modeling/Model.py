import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
import os
import dmbrl.modeling.NeuralNet as NeuralNet
os.environ['CUDA_VISIBLE_DEVICES'] = '2'


class Model():
    def __init__(self,learningRateG,learningRatex,image_dim,generator_dim,c = 100000):
        # print(str(cudaGPU))
        # exit()
        # device = torch.device("cuda:"+str(cudaGPU) if torch.cuda.is_available() else torch.device("cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        self.GNet = NeuralNet.Net(1,generator_dim,1)
        self.xNet = NeuralNet.Net(1, (1,1),1)
        # self.GNet = nn.DataParallel(self.GNet,device_ids = [3])
        self.GNet = self.GNet.to(device)
        self.xNet = self.xNet.to(device)
        # self.xNet = self.xNet.to(device)
        # self.GNet = nn.DataParallel(GNet)
        self.device =device
        self.optimG = Adam(self.GNet.parameters(), lr=learningRateG)
        self.optimx = Adam(self.xNet.parameters(), lr=learningRatex)

    def saveModels(self):
        torch.save(self.GNet.state_dict(),"GNet.pt")
        torch.save(self.xNet.state_dict(),"xNet.pt")
        torch.save(self.optimG.state_dict(),"optimG.pt")
        torch.save(self.optimx.state_dict(),"optimx.pt")

class MujocoModel():
    def __init__(self,learningRateG,learningRatex,generator_dim,c = 100000):
        # print(str(cudaGPU))
        # exit()
        # device = torch.device("cuda:"+str(cudaGPU) if torch.cuda.is_available() else torch.device("cpu"))
        device = torch.device("cuda" if torch.cuda.is_available() else torch.device("cpu"))
        self.GNet = NeuralNet.Net(1,generator_dim,1)
        self.xNet = NeuralNet.Net(1,(1,),1)
        # self.GNet = nn.DataParallel(self.GNet,device_ids = [3])
        self.GNet = self.GNet.to(device)
        # self.xNet = self.xNet.to(device)
        # self.GNet = nn.DataParallel(GNet)
        self.device =device
        self.optimG = Adam(self.GNet.parameters(), lr=learningRateG)
        self.optimx = Adam(self.xNet.parameters(), lr=learningRatex)

    def saveModels(self):
        torch.save(self.GNet.state_dict(),"GNet.pt")
        torch.save(self.xNet.state_dict(),"xNet.pt")
        torch.save(self.optimG.state_dict(),"optimG.pt")
        torch.save(self.optimx.state_dict(),"optimx.pt")