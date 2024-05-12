import torch
from torch.optim import Adam
from torch.distributions.normal import Normal
import torch.nn as nn
import torch.nn.functional as F

import dmbrl.modeling.NeuralNet as NeuralNet

class Model1D():
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
