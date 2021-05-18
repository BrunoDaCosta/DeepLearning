from torch import empty
import torch

from generate_data import *


def loss(v, t):
    #return (v - t).pow(2).sum()
    return (v - t).pow(2).mean()


def dloss(v, t):
    return 2 * (v - t)


class Module:

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError


class linear(Module):

    def __init__(self, nbinput, nboutput):
        super().__init__()
        self.w = empty(nboutput, nbinput).normal_(0, epsilon)
        self.b = empty(nboutput, 1).normal_(0, epsilon)

    def forward_pass(self, x):
        y = self.w.mm(x.t()).t() + self.b.view(1,-1)
        self.x = x
        # print(y.size())
        return y

    def backward_pass(self, dl_dy):
        dl_dx = self.w.t().mm(dl_dy.t()).t()

        self.dl_dw = 0
        self.dl_db = 0
        for i in range(dl_dy.shape[0]):
            self.dl_dw += self.x[i].view(-1,1).mm(dl_dy[i].t().view(1,-1)).t()
            self.dl_db += dl_dy[i].view(-1,1)

        #self.dl_dw = dl_dy.t().mm(self.x)
        #self.dl_db = dl_dy

        # print(self.dl_dw.size())
        # print(self.w.size())
        #print(self.dl_dw.size(), self.w.size())
        #print(self.dl_db.size(), self.b.size())
        self.w -= eta * self.dl_dw
        self.b -= eta * self.dl_db

        return dl_dx

    def param(self):
        return [(self.w, self.dl_dw * eta), (self.b, self.dl_db * eta)]


class tanh(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        return x.tanh()

    def backward_pass(self, dloss):
        return dloss * (4 * (x.exp() + x.mul(-1).exp()).pow(-2))

    def param(self):
        return []


class ReLu(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        self.x = x
        return x.relu()

    def backward_pass(self, dloss):
        return dloss * (self.x > 0).float()

    def param(self):
        return []


class sequential(Module):

    def __init__(self, *args):
        super().__init__()
        self.layers = args

    def forward_pass(self, input_data):
        for layer in self.layers:
            print(layer)
            input_data = layer.forward_pass(input_data)
        return input_data

    def backward_pass(self, dloss):
        for layer in reversed(self.layers):
            dloss = layer.backward_pass(dloss)
        return dloss

    def param(self):
        params = []
        for layer in self.layers:
            params += layer.param()
        return params


train_data, train_label, test_data, test_label = generate_data()
epsilon = 0.001
eta = 1e-1 / train_data.shape[0]
nbtests = 300


model = sequential(linear(2, 20), ReLu, linear(20,1))
for i in range(nbtests):
    yest = model.forward_pass(train_data)
    loss_der = dloss(yest, train_label)
    print(str(i) + " " + str(loss(train_label, model.forward_pass(train_data))))
    model.backward_pass(loss_der)

print(model.forward_pass(train_data))
print(loss(train_label, model.forward_pass(train_data)))
# print(train_label.t())