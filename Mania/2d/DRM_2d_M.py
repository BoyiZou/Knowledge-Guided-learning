import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os

class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
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

        # x = x*(xxx)*(1-xxx) + xxx + yyy*(xxx)*(1-xxx)*10
        x = x*(xxx)*(1-xxx) + xxx

        return x

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainnew(model,device,params,optimizer,scheduler):

    data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))

    xx = data[:,0].reshape(-1,1)
    yy = data[:,1].reshape(-1,1)
    data.requires_grad = True

    model.train()

    FF = open("loss.txt","w")
    EE = open("testloss.txt","w")
    for step in range(params["trainStep"]):

        if step%1000 == 0:
          test(model,device,params)

        u = model(data).reshape(-1,1)

        model.zero_grad()

        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

        dudx1 = dudx[:,0]
        dudx2 = dudx[:,1]

        loss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)
        loss1 = loss

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():

                print("Error at Step %s is %s."%(step+1,loss.cpu().detach().numpy()))
                file = open("loss.txt","a")
                file.write(str(loss.cpu().detach().numpy())+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:

            np.random.seed(step)
            data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)
            data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            xx = data[:,0].reshape(-1,1)
            yy = data[:,0].reshape(-1,1)
            data.requires_grad = True

        # if step%2000 == 1:
        #     test(model,device,params)

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        loss1.backward()

        optimizer.step()
        scheduler.step()

def exactnew(data,params):

    xx = data[:,0].reshape(-1,1)
    yy = data[:,1].reshape(-1,1)

    nexact = xx**(1/3)

    return nexact

def errorFunnew(nxt,target,params):

    error = (nxt-target)**2
    error = math.sqrt(torch.mean(error))

    return error

def test(model,device,params):

    plt.rcParams['font.size'] = 20

    Pix = 101
    x = torch.linspace(0, 1, Pix)
    y = torch.linspace(0, 1, Pix)
    TestData = torch.zeros([(Pix)**2,params["d"]]).to(device)

    X,Y = torch.meshgrid(x,y)

    XX = X.reshape(-1,1)
    YY = Y.reshape(-1,1)
    XX = XX.squeeze()
    YY = YY.squeeze()
    print(X.shape)
    TestData[:,0] = XX
    TestData[:,1] = YY

    TestData.requires_grad = True
    xx = TestData[:,0].reshape(-1,1)

    u = model(TestData).reshape(-1,1)

    model.zero_grad()

    dudx = torch.autograd.grad(u,TestData,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

    dudx1 = dudx[:,0].reshape(-1,1)
    dudx2 = dudx[:,1].reshape(-1,1)

    aaa = torch.max(torch.abs(dudx1))
    bbb = torch.max(torch.abs(dudx2))
    grad_inf_norm = torch.max(aaa,bbb)


    testloss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)
    print("Test Error at Step is %s."%(testloss))

    Pix2 = 96
    x2 = torch.linspace(0.005, 1, Pix2)
    y2 = torch.linspace(0.005, 1, Pix2)
    TestData2 = torch.zeros([(Pix2)**2,params["d"]]).to(device)

    X2,Y2 = torch.meshgrid(x2,y2)

    XX2 = X2.reshape(-1,1)
    YY2 = Y2.reshape(-1,1)
    XX2 = XX2.squeeze()
    YY2 = YY2.squeeze()
    print(X.shape)
    TestData2[:,0] = XX2
    TestData2[:,1] = YY2

    TestData2.requires_grad = True
    xx2 = TestData2[:,0].reshape(-1,1)

    u2 = model(TestData2).reshape(-1,1)

    model.zero_grad()

    dudx = torch.autograd.grad(u2,TestData2,grad_outputs=torch.ones_like(u2),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

    dudx1 = dudx[:,0].reshape(-1,1)
    dudx2 = dudx[:,1].reshape(-1,1)

    aaa2 = torch.max(torch.abs(dudx1))
    bbb2 = torch.max(torch.abs(dudx2))
    grad_inf_norm2 = torch.max(aaa2,bbb2)

    testloss2 = 0.995*torch.mean((u2**3-xx2)**2*(dudx[:,0])**6)
    print("Test Error2 at Step is %s."%(testloss2))

    file = open("testloss.txt","a")
    file.write(str(testloss.cpu().detach().numpy())+"\n")
    file = open("testloss2.txt","a")
    file.write(str(testloss2.cpu().detach().numpy())+"\n")

    dudx = dudx.cpu().detach().numpy()

    dudx1 = dudx[:,0]
    dudx2 = dudx[:,1]
    TestData = TestData.cpu().detach().numpy()
    u = u.cpu().detach().numpy()

    plt.figure()
    plt.imshow(u.reshape(Pix,Pix).T,cmap = 'viridis',origin='lower')
    x_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    y_labels = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
    plt.xticks(np.linspace(0, 100, len(x_labels)), x_labels)
    plt.yticks(np.linspace(0, 100, len(y_labels)), y_labels)

    plt.xticks(fontsize=22)

    plt.yticks(fontsize=22)

    cbar = plt.colorbar(shrink=0.7)
    cbar.ax.tick_params(labelsize=22)
    plt.show()

    data = torch.zeros((params['numQuad'],params["d"])).float().to(device)
    data[:,0] = torch.from_numpy(np.linspace(0,1,params['numQuad'],endpoint=True))
    data[:,1] = 1
    xx = data[:,0].reshape(-1,1)
    data.requires_grad=True

    u = model(data).reshape(-1,1)

    model.zero_grad()

    dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

    dudx = dudx.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    u = u.cpu().detach().numpy()

    plt.figure()
    plt.scatter(data[:,0],abs(data[:,0].reshape(-1,1)**(1/3)-u), marker='x',c='r',s=2)
    plt.xlabel('x', fontsize=22)
    plt.ylabel('$\epsilon$', fontsize=22)
    plt.show()

    plt.figure()
    plt.scatter(data[:,0], u, color='r', marker='x' ,s=0.45)
    plt.plot(data[:,0], data[:,0]**(1/3), color='k',linewidth=1)

    plt.legend(['Trained solution', 'Analytical solution'],markerscale=10,loc='upper left')
    plt.xlabel('x', fontsize=22)
    plt.ylabel('u', fontsize=22)
    plt.show()

    return 1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():
    # Parameters
    torch.manual_seed(22)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["radius"] = 1
    params["d"] = 2
    params["dd"] = 1
    params["bodyBatch"] = 10000
    params["lr"] = 0.006
    params["width"] = 50
    params["depth"] = 4
    params["numQuad"] = 1001
    params["trainStep"] = 30000
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["milestone"] = [1000,6000,11500,26000,34000,42000,48000,54000,58000,59000,59900,60000,67000,72000,78000,85000,90000,95000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = RitzNet(params).to(device)
    # model.apply(initWeights)
    torch.save(model.state_dict(),"first_model.pt")
    print("Generating network costs %s seconds."%(time.time()-startTime))

    optimizer = torch.optim.Adam(model.parameters(),lr=params["lr"],weight_decay=params["decay"])
    scheduler = MultiStepLR(optimizer,milestones=params["milestone"],gamma=params["gamma"])

    startTime = time.time()
    trainnew(model,device,params,optimizer,scheduler)
    print("Training costs %s seconds."%(time.time()-startTime))

    model.eval()
    testError = test(model,device,params)
    print("The test error (of the last model) is %s."%testError)
    print("The number of parameters is %s,"%count_parameters(model))

    torch.save(model.state_dict(),"last_model.pt")



if __name__=="__main__":
    main()

#     model = RitzNet(params).to(device)
#     model.load_state_dict(torch.load('/content/last_model_2.pt'))

#     test(model,device,params)

# if __name__=="__main__":
#     main()