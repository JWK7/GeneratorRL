import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F
import os
import dmbrl.modeling.NeuralNet as NeuralNet
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'


class Model1D():
    def __init__(self,learningRateG,learningRatex,image_dim,generator_dim,c = 100000,cudaGPU = 1):
        # print(str(cudaGPU))
        # exit()
        device = torch.device("cuda:"+str(cudaGPU) if torch.cuda.is_available() else torch.device("cpu"))
        self.GNet = NeuralNet.Net(image_dim,generator_dim,c)
        self.xNet = NeuralNet.Net(image_dim,(1,),c)
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
    

class Model2D():
    def __init__(self,learningRateG,learningRatex,image_dim,generator_dim,c = 100000):
        self.GNet = NeuralNet.Net(image_dim,generator_dim,c)
        self.xNet = NeuralNet.Net(image_dim,(1,),c)
        self.optimG = Adam(self.GNet.parameters(), lr=learningRateG)
        self.optimx = Adam(self.xNet.parameters(), lr=learningRatex)

    def saveModels(self):
        torch.save(self.GNet.state_dict(),"GNet.pt")
        torch.save(self.xNet.state_dict(),"xNet.pt")
        torch.save(self.optimG.state_dict(),"optimG.pt")
        torch.save(self.optimx.state_dict(),"optimx.pt")

class Model3D():
    def __init__(self,learningRateG,learningRatex,image_dim,generator_dim,c = 1000000):
        self.GNet = NeuralNet.Net(image_dim,generator_dim,c)
        self.xNet = NeuralNet.Net(image_dim,(1,),c)
        self.optimG = Adam(self.GNet.parameters(), lr=learningRateG)
        self.optimx = Adam(self.xNet.parameters(), lr=learningRatex)

    def saveModels(self):
        torch.save(self.GNet.state_dict(),"GNet.pt")
        torch.save(self.xNet.state_dict(),"xNet.pt")
        torch.save(self.optimG.state_dict(),"optimG.pt")
        torch.save(self.optimx.state_dict(),"optimx.pt")
