from torch import empty
import torch

from generate_data import *


def loss(v, t):
    return (v - t).pow(2).sum()


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
        self.w = empty(nbinput, nboutput).normal_(0, epsilon)
        self.b = empty(nboutput).normal_(0, epsilon)

    def forward_pass(self, x):
        y = x.mm(self.w) + self.b
        self.x = x
        return y

    def backward_pass(self, dl_dy):
        dl_dx = self.w.mm(dl_dy.t())
        self.dl_dw = dl_dy.t().mm(self.x)
        self.dl_db = dl_dy

        self.w = self.w - eta * self.dl_dw
        self.b = self.b - eta * self.dl_db

        return dl_dx

    def param(self):
        return [(self.w, self.dl_dw * eta), (self.b, self.dl_db * eta)]


class tanh(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        return x.tanh()

    def backward_pass(self, dloss):
        return dloss * (4 * (x.exp() + x.mul(-1).exp()).pow(-2))

    def param(self):
        return []


class relu(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        return x.relu()

    def backward_pass(self, dloss):
        return dloss * (x > 0).float()

    def param(self):
        return []


class sequential(Module):

    def __init__(self, *args):
        super().__init__()
        self.layers = args

    def forward_pass(self, input_data):
        for layer in self.layers:
            input_data = layer.forward_pass(input_data)
        return input_data

    def backward_pass(self, dloss):
        for layer in self.layers: # wrong way
            dloss = layer.backward_pass(dloss)
        return dloss

    def param(self):
        params = []
        for layer in self.layers:
            params += layer.param()
        return params


train_data, train_label, test_data, test_label = generate_data_lin()
epsilon = 0.001
"""
model = sequential(linear(1,2))

input_data = torch.tensor([1.])
print(model.param())
"""
eta = 1e-1 / train_data.shape[0]
nbtests = 100

model = sequential(linear(2,1))

for i in range(nbtests):
    yest = model.forward_pass(train_data)
    loss = dloss(yest, train_label)
    #print(yest, loss, train_label)
    #print(loss.sum())
    model.backward_pass(loss)

print(model.forward_pass(train_data))
# print(train_label.t())
