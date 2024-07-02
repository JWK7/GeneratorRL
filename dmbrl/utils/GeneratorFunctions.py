import torch
import torch.nn as nn
def UpdateGradienceG(net,data):
    G = net.GNet((data[2].unsqueeze(-1)).to(net.device))
    # x = net.xNet((data[2].unsqueeze(-1)).to(net.device)).detach()
    # xG = x * G
    xG = G
    I0 = torch.zeros((data.shape[1],2,2))
    I0[:,0,1] = -data[0]
    I0[:,1,0] = data[0]
    I0 = I0.to(net.device)

    deltaI = torch.zeros((data.shape[1],2,2))
    deltaI[:,0,1] = -data[2]
    deltaI[:,1,0] = data[2]
    deltaI = deltaI.to(net.device)
    xGI0 = torch.matmul(xG,I0)


    print(xGI0)
    # print(G)
    print(deltaI)
    # exit()

    loss = nn.MSELoss()
    net.optimG.zero_grad()
    grad = loss(xGI0.squeeze(),(deltaI).squeeze().to(net.device))
    grad.backward()
    # print(grad)
    net.optimG.step()

    return grad.detach().to('cpu')

def UpdateG(params,data):
    # sim_data = params.tool_cfg.sample()
    for i,net in enumerate(params.tool_cfg.nn):
        if i < 1:
            params.log_cfg.Gloss.append(UpdateGradienceG(net,data[:,:,i]))

def UpdateGradienceX(net,data):
    G = net.GNet((data[2].unsqueeze(-1)).to(net.device)).detach()
    x = net.xNet((data[2].unsqueeze(-1)).to(net.device))
    xG = x * G
    I0 = torch.zeros((data.shape[1],2,2))
    for i in range(data.shape[1]): I0[i] = torch.eye(2)
    I0 = data[0] * I0
    I0 = I0.to(net.device)

    deltaI = torch.ones((data.shape[1],2,2))
    deltaI[:,0,1] = -data[2]
    deltaI[:,1,0] = data[2]
    deltaI = deltaI.to(net.device)
    xGI0 = torch.matmul(xG,I0)

    loss = nn.MSELoss()
    net.optimx.zero_grad()
    grad = loss(xGI0.squeeze(),(deltaI).squeeze().to(net.device))
    grad.backward()
    net.optimx.step()
    return grad.detach().to('cpu')

def UpdateX(params,data):
    for i,net in enumerate(params.tool_cfg.nn):
        if i < 7:
            params.log_cfg.xloss.append(UpdateGradienceX(net,data[:,:,i]))

# def Update(params):
#     I0 = params.tool_cfg.sample()
#     print(I0.shape)
#     print(I0[:,:,1])