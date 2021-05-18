from torch import empty
from generate_data import *

def loss(v, t):
    return (v - t).pow(2).sum()

def dloss(v, t):
    return 2 * (v - t)

class Module:
    def __init__(self):
        pass
    def forward(self , * input):
        raise NotImplementedError

    def backward(self , * gradwrtoutput):
        raise NotImplementedError

    def param(self):
        return ["jaj"]

class linear(Module):
    def __init__(self, nbinput, nboutput):
        super().__init__()
        self.w = empty(nbinput, nboutput).normal_(0, epsilon)
        self.b = empty(nboutput).normal_(0, epsilon)

    def forward_pass(self,x):
        y = self.w.mv(x) + self.b
        return y

    def backward_pass(self, x, dl_dy):
        dl_dx = self.w.t().mv(dl_dy)
        dl_dw = dl_dy.view(-1, 1).mm(x.view(1, -1))
        dl_db = dl_dy

        self.w = self.w - eta * dl_dw
        self.b = self.b - eta * dl_db

        return dl_dx

class tanh(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        return x.tanh()

    def backward_pass(self, dloss):
        return dloss * (4 * (x.exp() + x.mul(-1).exp()).pow(-2))

class relu(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self):
        return x.relu()

    def backward_pass(self, dloss):
        return dloss * (x > 0).float()


train_data, train_label, test_data, test_label = generate_data()
epsilon = 0.001
eta = 1e-6
a = linear(1,2)
print(a.param())