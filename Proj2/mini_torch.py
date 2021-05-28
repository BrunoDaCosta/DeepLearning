import math
from torch import empty


def LossMSE(y_est, y):
    return (y_est - y).pow(2).sum()


def dLossMSE(y_est, y):
    return 2 * (y_est - y)


class Module:

    def __init__(self):
        pass

    def forward(self, *input):
        raise NotImplementedError

    def backward(self, *gradwrtoutput):
        raise NotImplementedError

    def param(self):
        raise NotImplementedError


class Linear(Module):

    def __init__(self, nbinput, nboutput, epsilon=0.3):
        super().__init__()
        self.w = empty(nboutput, nbinput).normal_(0, epsilon)
        self.b = empty(nboutput).normal_(0, epsilon)
        self.dl_dw = empty(self.w.size()).zero_()
        self.dl_db = empty(self.b.size()).zero_()

    def forward(self, x):
        self.x = x
        return self.w.mv(x) + self.b

    def backward(self, dl_dy):
        dl_dx = self.w.t().mv(dl_dy)
        self.dl_dw.add_((self.x.view(-1, 1).mm(dl_dy.view(-1, 1).t())).t())
        self.dl_db.add_(dl_dy)
        return dl_dx

    def param(self):
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]


class Tanh(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return x.tanh()

    def backward(self, dloss):
        return dloss * (4 * (self.x.exp() + self.x.mul(-1).exp()).pow(-2))

    def param(self):
        return []


class Sigmoid(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        return 1 / (1 + math.exp(-x))

    def backward(self, dloss):
        gv = 1 / (1 + math.exp(-self.x))
        return gv * (1 - gv) * dloss

    def param(self):
        return []


class ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        x[x < 0] = 0
        return x

    def backward(self, dloss):
        return dloss * (self.x > 0).float()

    def param(self):
        return []


class leaky_ReLU(Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        self.x = x
        x[x < 0] *= 0.01
        return x

    def backward(self, dloss):
        self.x[self.x <= 0] = 0.01
        self.x[self.x > 0] = 1
        return dloss * self.x

    def param(self):
        return []


class Sequential(Module):

    def __init__(self, *args, eta):
        super().__init__()
        self.layers = args
        self.eta = eta

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer.forward(input_data)
        return input_data

    def backward(self, dloss):
        for layer in reversed(self.layers):
            dloss = layer.backward(dloss)
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
                weight, dw = par[0]
                weight -= dw * self.eta
                bias, db = par[1]
                bias -= db * self.eta

    def zero_grad(self):
        for layer in self.layers:
            par = layer.param()
            if par:
                _, dw = par[0]
                dw.zero_()
                _, db = par[1]
                db.zero_()


def classify(result, objective):
    result = result >= 0.5
    errors = result != objective
    return errors.sum()
