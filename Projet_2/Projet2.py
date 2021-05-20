from torch import empty
import torch

from generate_data import *


def MSEloss(v, t):
    return (v - t).pow(2).sum()


def MSEdloss(v, t):
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
        self.b = empty(nboutput).normal_(0, epsilon)
        self.dl_dw = empty(self.w.size()).zero_()
        self.dl_db = empty(self.b.size()).zero_()
        #print("debut")
        #print(self.w)
        #print(self.b)
        #print("###")

    def forward_pass(self, x):
        self.x = x
        #print("x, x_next")
        #print(x, self.w.mv(x) + self.b)
        return self.w.mv(x) + self.b

    def backward_pass(self, dl_dy):
        #print("dl_dy")
        #print(dl_dy)
        #print("w, b")
        #print(self.w)
        #print(self.b)
        #print("db")
        #print(self.dl_db)

        dl_dx = self.w.t().mv(dl_dy)

        self.dl_dw.add_(dl_dy.view(-1, 1).mm(self.x.view(1, -1)))
        self.dl_db.add_(dl_dy)
        #print("x")
        #print(self.x)
        #print("dw")
        #print(self.dl_dw)
        #print("db")
        #print(self.dl_db)
        #self.w = self.w - eta * self.dl_dw
        #self.b = self.b - eta * self.dl_db
        #print("w, b")
        #print(self.w)
        #print(self.b)
        #print("###")
        return dl_dx

    def param(self):
        return [(self.w, self.dl_dw * eta), (self.b, self.dl_db * eta)]


class tanh(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        self.x = x
        return x.tanh()

    def backward_pass(self, dloss):
        return dloss * (4 * (self.x.exp() + self.x.mul(-1).exp()).pow(-2))

    def param(self):
        return []


class relu(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        self.x = x
        x[x < 0] = 0
        return x

    def backward_pass(self, dloss):
        #print(dloss)
        #print(dloss * (self.x > 0).float())
        return dloss * (self.x > 0).float()

    def param(self):
        return []

class leaky_relu(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        self.x = x
        x[x < 0] *= 0.01
        return x

    def backward_pass(self, dloss):
        #print(dloss)
        #print(dloss * (self.x > 0).float())
        self.x[self.x < 0] = 0.01
        self.x[self.x > 0] = 1
        return dloss * self.x

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
        for layer in reversed(self.layers):
            #print("dloss")
            #print(dloss)
            dloss = layer.backward_pass(dloss)
        return dloss

    def param(self):
        params = []
        for layer in self.layers:
            params += layer.param()
        return params

    def step(self):
        for layer in self.layers:
            par = layer.param()
            if par:
                weight = par[0]
                weight, dw = weight
                #print("weight")
                #print(weight)
                #print(dw)
                weight -= dw
                bias = par[1]
                bias, db = bias
                #print("bias")
                #print(bias)
                #print(db)
                bias -= db
    
    def zero_grad(self):
        for layer in self.layers:
            par = layer.param()
            if par:
                weight = par[0]
                weight, dw = weight
                dw.zero_()
                bias = par[1]
                bias, db = bias
                db.zero_()


train_input, train_label, test_input, test_label = generate_data_lin()
mean, std = train_input.mean(), train_input.std()
train_input.sub_(mean).div_(std)
test_input.sub_(mean).div_(std)

#train_label.zero_()
#train_label += 1
epsilon = 0.1
"""
model = sequential(linear(1,2))

input_data = torch.tensor([1.])
print(model.param())
"""

nb_input = train_input.size(0)
eta = 1e-2 / nb_input
nbtests = 50

model = sequential(linear(2,25), leaky_relu(), linear(25, 1), tanh())
par = model.param()
#print(par)
#print(train_label.size())
#print(train_input)
for i in range(nbtests):
    mod_loss = 0
    model.zero_grad()
    for n in range(nb_input):
        yest = model.forward_pass(train_input[n])
        #print(train_input[n], yest)
        mod_loss += MSEloss(yest, train_label[n])
        loss = MSEdloss(yest, train_label[n])
        #print(yest, loss, train_label[n])
        #print(loss)
        model.backward_pass(loss)
    model.step()
    print("MSE")
    print(mod_loss)
    print("###")

par = model.param()
#print(par)
for n in range(nb_input):
    xest = model.forward_pass(train_input[n])
    print(xest)
print(train_label.t())