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
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"]))

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"])

    def forward(self, x):

        xxx = x[:,0].reshape(-1,1)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3
            x = x_temp+x

        x = self.linearOut(x)
        x = x*(xxx)*(1-xxx) + xxx

        return x

def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainnew(model,device,params,optimizer,scheduler):
    model.train()
    data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)
    data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
    xx = data[:,0].reshape(-1,1)
    data.requires_grad = True

    FF = open("loss.txt","w")
    EE = open("testloss.txt","w")
    for step in range(params["trainStep"]):
        u = model(data).reshape(-1,1)

        model.zero_grad()
        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]
        loss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():

                print("Error at Step %s is %s."%(step+1,loss))
                file = open("loss.txt","a")
                file.write(str(loss)+"\n")

        if step%params["sampleStep"] == params["sampleStep"]-1:

            np.random.seed(step)
            data = torch.zeros((params["bodyBatch"],params["width"])).float().to(device)


            data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            xx = data[:,0].reshape(-1,1)
            data.requires_grad = True

        if step%5000 == 1:
            test(model,device,params)

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        if step < 4000:
          loss = torch.mean((u.reshape(-1,1)-data[:,0].reshape(-1,1)**(1/3))**2)

        loss.backward()

        optimizer.step()
        scheduler.step()

def exactnew(data,params):

    xx = data[:,0].reshape(-1,1)
    nexact = xx**(1/3)

    return nexact

def errorFunnew(nxt,target,params):

    error = (nxt-target)**2
    error = math.sqrt(torch.mean(error))

    return error

def test(model,device,params):

    data = torch.zeros((params['numQuad'],params["width"])).float().to(device)
    data[:,0] = torch.from_numpy(np.linspace(0,1,params['numQuad'],endpoint=True))
    xx = data[:,0].reshape(-1,1)
    data.requires_grad=True

    u = model(data).reshape(-1,1)

    model.zero_grad()

    dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

    testloss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)

    print("Test Error at Step is %s."%(testloss))
    testloss2 = torch.mean((u[5:,:]**3-xx[5:,:])**2*(dudx[5:,0])**6)
    print("Test Error2 at Step is %s."%(testloss2))
    file = open("testloss.txt","a")
    file.write(str(testloss.cpu().detach().numpy())+"\n")

    dudx = dudx.cpu().detach().numpy()
    data = data.cpu().detach().numpy()
    u = u.cpu().detach().numpy()

    plt.rcParams['font.size'] = 20

    ppp = plt.figure()
    plt.scatter(data[:,0],u,c='y',s=0.05)
    plt.show()

    plt.figure()
    plt.scatter(data[:,0],data[:,0]**(1/3),c='y',s=0.05)
    plt.show()

    plt.figure()
    plt.scatter(data[:,0],dudx[:,0],c='y',s=0.05)
    plt.show()

    plt.figure()
    plt.scatter(data[:,0],abs(data[:,0].reshape(-1,1)**(1/3)-u),c='r', marker='x',s=0.45)
    plt.xlabel('x', fontsize=22)
    plt.ylabel('$\epsilon$', fontsize=22)
    plt.savefig('DRM Mania error.eps',  format='eps')
    plt.savefig('DRM Mania error.png',  format='png')
    plt.show()

    plt.figure()
    plt.scatter(data[:,0], u, color='r', marker='x',s=0.45)
    plt.plot(data[:,0], data[:,0]**(1/3), color='k',linewidth=1)

    plt.legend(['Trained solution', 'Analytical solution'],markerscale=10,loc='upper left')
    plt.xlabel('x', fontsize=22)
    plt.ylabel('u', fontsize=22)
    plt.savefig('DRM Mania TT.eps',  format='eps')
    plt.savefig('DRM Mania TT.png',  format='png')
    plt.show()

    return 1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():

    torch.manual_seed(22)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = dict()
    params["d"] = 120
    params["dd"] = 1
    params["bodyBatch"] = 10000
    params["lr"] = 0.016
    params["width"] = 120
    params["depth"] = 8
    params["numQuad"] = 1000
    params["trainStep"] = 80000
    params["writeStep"] = 50
    params["sampleStep"] = 10
    params["step_size"] = 5000
    params["milestone"] = [1000,1800,2500,5000,8000,14000,14500,28000,31000,34000,37000,40000,42000,45000,48000,52000,54000,56000,58000,59000,60000,67000,72000,78000]
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = RitzNet(params).to(device)
    # model.apply(initWeights)
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
#     model.load_state_dict(torch.load('/content/last_model (24).pt'))

#     test(model,device,params)

# if __name__=="__main__":
#     main()