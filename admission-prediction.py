import torch
import torch.nn as nn
import matplotlib.pyplot as plt

x = torch.tensor(([420,60],[510,92],[488,80],[533,90],[455,79],[512,78],[485,95],[529,65],[449,82],[540,100]), dtype=torch.float)
y = torch.tensor(([10],[80],[40],[90],[30],[50],[65],[55],[49],[100]), dtype=torch.float)
xPredicted = torch.tensor(([500, 99]), dtype=torch.float)

x_max, _ = torch.max(x, 0) # search column by column, if 1, row by row
xPredicted_max, _ =torch.max(xPredicted, 0)

x = torch.div(x, x_max)
xPredicted = torch.div(xPredicted, x_max)
y = y/100

class Neural_Network(nn.Module):
    def __init__(self,):
        super(Neural_Network, self).__init__()
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        # weights
        self.w1 = torch.randn(self.inputSize, self.hiddenSize)
        self.w2 = torch.randn(self.hiddenSize, self.outputSize)

        self.z2 = 0

    def sigmoid(self,s):
        return 1 / (1+torch.exp(-s))

    def sigmoidPrime(self, s):
        return s*(1-s)

    def forward(self, x):
        z1 = torch.matmul(x,self.w1)
        self.z2 = self.sigmoid(z1)
        z3 = torch.matmul(self.z2, self.w2)
        output = self.sigmoid(z3)
        return output

    def backward(self, x, y, o):
        o_error = y - o # error in output
        o_delta = o_error * self.sigmoidPrime(o)
        z2_error = torch.matmul(o_delta, torch.t(self.w2))
        z2_delta = z2_error * self.sigmoidPrime(self.z2)
        self.w1 += torch.matmul(torch.t(x), z2_delta)
        self.w2 += torch.matmul(torch.t(self.z2), o_delta)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x,y,o)

    def saveWeights(self, model):
        torch.save(model, "NN")

    def predict(self):
        print("Predicted data based on trained weights: ")
        print("Input (scaled): \n" + str(xPredicted))
        print("Output: \n" + str(self.forward(xPredicted)))

if __name__ == '__main__':
    NN = Neural_Network()
    l = []
    for i in range(1000):
        loss = torch.mean((y - NN(x))**2).detach().item()
        print("#" + str(i) + "Loss: " + str(loss))
        l.append(loss)
        NN.train(x,y)
    NN.saveWeights(NN)
    NN.predict()
    plt.plot(l)
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
