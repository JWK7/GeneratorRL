import torch
import torch.nn as nn
def UpdateGradienceG(net,data):
    G = net.GNet((data[2].unsqueeze(-1)).to(net.device))
    x = net.xNet((data[2].clone().unsqueeze(-1)).to(net.device)).detach()
    xG = x * G
    I0 = torch.zeros((data.shape[1],2,2))
    I0[:,0,1] = -data[0]
    I0[:,1,0] = data[0]

    deltaI = torch.zeros((data.shape[1],2,2))
    deltaI[:,0,1] = -data[2]
    deltaI[:,1,0] = data[2]
    xGI0 = torch.matmul(xG,I0)

    loss = nn.MSELoss()
    net.optimG.zero_grad()
    grad = loss(xGI0.squeeze(),(deltaI).squeeze().to(net.device))
    net.optimG.step()
    pass

def UpdateG(params,data):
    # sim_data = params.tool_cfg.sample()
    for i,net in enumerate(params.tool_cfg.nn):
        UpdateGradienceG(net,data[:,:,i])
    pass

def UpdateGradienceX(net,data):
    G = net.GNet((data[2].unsqueeze(-1)).to(net.device)).detach()
    x = net.xNet((data[2].clone().unsqueeze(-1)).to(net.device))
    xG = x * G
    I0 = torch.zeros((data.shape[1],2,2))
    I0[:,0,1] = -data[0]
    I0[:,1,0] = data[0]

    deltaI = torch.zeros((data.shape[1],2,2))
    deltaI[:,0,1] = -data[2]
    deltaI[:,1,0] = data[2]
    xGI0 = torch.matmul(xG,I0)

    loss = nn.MSELoss()
    net.optimx.zero_grad()
    grad = loss(xGI0.squeeze(),(deltaI).squeeze().to(net.device))
    net.optimx.step()
    pass

def UpdateX(params,data):
    for i,net in enumerate(params.tool_cfg.nn):
        UpdateGradienceX(net,data[:,:,i])
    pass

# def Update(params):
#     I0 = params.tool_cfg.sample()
#     print(I0.shape)
#     print(I0[:,:,1])