import torch.nn as nn
import torch.nn.functional as F
import torch

class Net(nn.Module):

    def __init__(self,input_dim,output_dim,c):
        super(Net, self).__init__()
        if type(input_dim) is tuple:
            input_dim = self.get_flattened_dim(input_dim)

        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 1000,bias=False)
        self.fc2 = nn.Linear(1000, 1000,bias=False)
        self.fc3 = nn.Linear(1000, self.get_flattened_dim(output_dim),bias=False)
        self.act = torch.nn.Sigmoid()
        self.c = c


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape((x.shape[0],)+self.output_dim)
        return x/self.c

    def get_flattened_dim(self,dim):
        if len(dim) == 1:
            return dim[0]
        return dim[0] * self.get_flattened_dim(dim[1:])


class ReparamNet(nn.Module):

    def __init__(self,input_dim,output_dim,c):
        super(Net, self).__init__()
        if type(input_dim) is tuple:
            input_dim = self.get_flattened_dim(input_dim)

        self.output_dim = output_dim
        self.fc1 = nn.Linear(input_dim, 1000,bias=False)
        self.fc2 = nn.Linear(1000, 1000,bias=False)
        self.fc3 = nn.Linear(1000, self.get_flattened_dim(output_dim),bias=False)
        self.act = torch.nn.Sigmoid()
        self.c = c


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).reshape((x.shape[0],)+self.output_dim)
        return x/self.c

    def get_flattened_dim(self,dim):
        if len(dim) == 1:
            return dim[0]
        return dim[0] * self.get_flattened_dim(dim[1:])