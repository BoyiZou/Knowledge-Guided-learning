import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os
import mpl_toolkits.mplot3d as p3d

class RitzNet(torch.nn.Module):
    def __init__(self, params, device):
        super(RitzNet, self).__init__()
        self.params = params
        self.device = device
        self.linearIn = nn.Linear(self.params["d"], self.params["width"])
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):

        xxx = x[:,0].reshape(-1,1)
        yyy = x[:,1].reshape(-1,1)

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x = (1-xxx)*xxx*x + xxx

        return x


def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.xavier_zeros_(m.bias)

def trainnew(model,device,params,optimizer,scheduler,optimizer2,scheduler2):

    model.train()

    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"])) + 1.5

    y_y_1 =  data[:,1].reshape(-1,1)/(data[:,1].reshape(-1,1)-1)

    data.requires_grad = True

    Pix = 100
    x = torch.arange(0,1+1/Pix,1/Pix)
    y = torch.arange(1.5,2.5+1/Pix,1/Pix)
    TestData = torch.zeros([(Pix+1)**2,params["d"]]).to(device)

    X,Y = torch.meshgrid(x,y)

    XX = X.reshape(-1,1)
    YY = Y.reshape(-1,1)
    XX = XX.squeeze()
    YY = YY.squeeze()

    xxx = XX
    yyy = YY
    print(X.shape)
    TestData[:,0] = XX
    TestData[:,1] = YY

    exact_sol = (xxx**((yyy-1)/(yyy))).reshape(Pix+1,Pix+1).T

    TestData.requires_grad = True

    for step in range(params["trainStep"]):

      if step%params["plotStep"] == 0:
        Test_u = model(TestData)
        Test_u = Test_u.cpu().detach().numpy().reshape(Pix+1,Pix+1).T

        plt.imshow(Test_u,cmap = 'viridis',origin='lower')
        cb = plt.colorbar(shrink=0.7)
        plt.title('Training results')
        plt.show()

        plt.figure()

        plt.imshow(exact_sol,cmap = 'viridis',origin='lower')
        cb = plt.colorbar(shrink=0.7)
        plt.title('exact')
        plt.show()


        plt.imshow(Test_u - exact_sol.cpu().numpy(),cmap = 'viridis',origin='lower')
        cb = plt.colorbar(shrink=0.7)
        plt.title('error')
        plt.show()


        error = Test_u - exact_sol.cpu().numpy()
        error = np.sqrt(np.mean((error)**2)/np.mean((exact_sol.cpu().numpy())**2))
        print("Error at Step %s is %s."%(step+1, error))

      u = model(data).reshape(-1,1)
      model.zero_grad()

      grad_u = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

      du_dx = grad_u[:,0].reshape(-1,1)
      du_dy = grad_u[:,1].reshape(-1,1)

      Loss = 66*0.35433531021*torch.mean(y_y_1**14*torch.abs(u)**((14-3*data[:,1].reshape(-1,1))/(data[:,1].reshape(-1,1)-1))*(torch.abs(u)**y_y_1 -data[:,0].reshape(-1,1))**2*du_dx**14)

      if step%params["sampleStep"] == params["sampleStep"]-1:

        np.random.seed(step)

        data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
        data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
        data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"])) + 1.5

        y_y_1 =  data[:,1].reshape(-1,1)/(data[:,1]-1).reshape(-1,1)

        data.requires_grad = True

      if step%params["writeStep"] == params["writeStep"]-1:

        print("Loss at Step %s is %s."%(step+1, Loss ))

      if  step%2 > 0:

        aaa = torch.max(torch.abs(du_dx))
        bbb = torch.max(torch.abs(du_dy))

        loss_adv = - torch.max(aaa,bbb)
        loss_adv.backward()

        optimizer.step()
        scheduler.step()

      else:

        Loss.backward()

        optimizer.step()
        scheduler.step()

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()

    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 1
    params["bodyBatch"] = 40000
    params["lr"] = 0.0016
    params["lr2"] = 0.0001
    params["width"] = 20
    params["depth"] = 4
    params["trainStep"] = 10000
    params["writeStep"] = 50
    params["plotStep"] = 500
    params["sampleStep"] = 100
    params["lambda"] = 1.8
    params["milestone"] = [1500,2500,4000,7000,11000,15000,19000,25000,30000,35000,39000]
    params["milestone2"] = [4000,7000,11000,15000,19000,25000,30000,35000,39000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = RitzNet(params,device).to(device)
    print("Generating network costs %s seconds."%(time.time()-startTime))

    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

    optimizer2 = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler2 = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

    startTime = time.time()
    trainnew(model,device,params,optimizer,scheduler,optimizer2,scheduler2)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")

if __name__=="__main__":
    main()