import numpy as np
import math, torch, time
import torch.nn.functional as F
from torch.optim.lr_scheduler import MultiStepLR, StepLR, MultiplicativeLR
import torch.nn as nn
import matplotlib.pyplot as plt
import sys, os

# Network structure
class RitzNet(torch.nn.Module):
    def __init__(self, params):
        super(RitzNet, self).__init__()
        self.params = params
        self.linearIn = nn.Linear(self.params["d"], self.params["width"]) # Input layer
        self.linear = nn.ModuleList()
        for _ in range(params["depth"]):
            self.linear.append(nn.Linear(self.params["width"], self.params["width"])) # Hidden layers

        self.linearOut = nn.Linear(self.params["width"], self.params["dd"]) # Output layer

    def forward(self, x):

        xxx = x[:,0].reshape(-1,1)

        x = self.linearIn(x)
        for layer in self.linear:
            x_temp = (F.tanh(layer(x)))**3 # Activation
            x = x_temp+x                   # Residual connection

        x = self.linearOut(x)   
        x = x*(xxx)*(1-xxx) + xxx # Modification to strictly satisfy the boundary conditions (The additional summation term can be any function that strictly 
                                  # satisfies the left and right boundary conditions. Since x is the most intuitive and straightforward choice, it is used here.
                                  #  Using other functions (e.g. x^2) would yield similar training results.)

        return x

# Initialization of parameters
def initWeights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_normal_(m.weight)
        torch.nn.init.zeros_(m.bias)

def trainnew(model,device,params,optimizer,scheduler):
 
    # Generate training data
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
        # Computing the gradients
        dudx = torch.autograd.grad(u,data,grad_outputs=torch.ones_like(u),retain_graph=True,create_graph=True,only_inputs=True,allow_unused=True)[0]

        dudx1 = dudx[:,0]
        dudx2 = dudx[:,1]

        # Training process of KGL: 4 steps for original functional and 1 step for the guiding term.
        if step < 40000:

          if step%5 >3:
            aaa = torch.max(torch.abs(dudx1))
            bbb = torch.max(torch.abs(dudx2))
            grad_inf_norm = torch.max(aaa,bbb)

            loss1 = -grad_inf_norm
          else:
            loss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)
            loss1 = loss
        else:
          loss = torch.mean((u**3-xx)**2*(dudx[:,0])**6)
          loss1 = loss

        if step%params["writeStep"] == params["writeStep"]-1:
            with torch.no_grad():

                print("Error at Step %s is %s."%(step+1,loss.cpu().detach().numpy()))
                file = open("loss.txt","a")
                file.write(str(loss.cpu().detach().numpy())+"\n")

        # Data shuffling as used in deep Ritz method
        if step%params["sampleStep"] == params["sampleStep"]-1:

            np.random.seed(step)
            data = torch.zeros((params["bodyBatch"],params["d"])).float().to(device)


            data[:,0] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            data[:,1] = torch.from_numpy(np.random.rand(params["bodyBatch"]))
            xx = data[:,0].reshape(-1,1)
            yy = data[:,0].reshape(-1,1)
            data.requires_grad = True

        if 10*(step+1)%params["trainStep"] == 0:
            print("%s%% finished..."%(100*(step+1)//params["trainStep"]))

        # backpropagation and scheduling for the learning rate
        loss1.backward()

        optimizer.step()
        scheduler.step()

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
    plt.scatter(data[:,0],abs(data[:,0].reshape(-1,1)**(1/3)-u),c='b',s=2)
    plt.xlabel('x', fontsize=22)
    plt.ylabel('$\epsilon$', fontsize=22)
    plt.show()

    plt.figure()
    plt.scatter(data[:,0], u, color='b',s=0.45)
    plt.plot(data[:,0], data[:,0]**(1/3), color='k',linewidth=1)

    plt.legend(['Trained solution', 'Analytical solution'],markerscale=10,loc='upper left')
    plt.xlabel('x', fontsize=22)
    plt.ylabel('u', fontsize=22)
    plt.show()

    return 1

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

def main():

    torch.manual_seed(22)
    np.random.seed(1)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    #parameters
    params = dict()
    params["d"] = 2  # Input dimension
    params["dd"] = 1  # Output dimension
    params["bodyBatch"] = 10000  # Nubmer of training data
    params["lr"] = 0.01  # learning rate. Here, we can use the same learning rate for both processes.
    params["width"] = 50  # width of the network 
    params["depth"] = 4   # Number of hidden layers of the network 
    params["numQuad"] = 1001 # Number of test point (1d)
    params["trainStep"] = 50000 # Total training step
    params["writeStep"] = 50 
    params["sampleStep"] = 10
    params["milestone"] = [1000,1800,2000,2100,2200,14500,16000,18000,19500,21000,25000,29000,34000,42000,45000,48000,52000,54000,56000,58000,59000,60000,67000,72000,78000] 
    params["gamma"] = 0.5
    params["decay"] = 0.0001

    startTime = time.time()
    model = RitzNet(params).to(device)
    torch.save(model.state_dict(),"first_model.pt")
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
#     model.load_state_dict(torch.load('/content/last_model (26).pt'))

#     test(model,device,params)

# if __name__=="__main__":
#     main()
