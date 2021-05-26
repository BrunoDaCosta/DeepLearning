from torch import empty
import math


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

    def __init__(self, nbinput, nboutput, epsilon = 0.3):
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
        #print(self.x)
        #print(dl_dy)
        self.dl_dw.add_((self.x.view(-1,1).mm(dl_dy.view(-1,1).t())).t())
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
        return [(self.w, self.dl_dw), (self.b, self.dl_db)]

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

class sigmoid(Module):

    def __init__(self):
        super().__init__()

    def forward_pass(self, x):
        self.x = x
        return 1/(1+math.exp(-x))

    def backward_pass(self, dloss):
        gv = 1 / (1 + math.exp(-self.x))
        return gv*(1 - gv)*dloss

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
        self.x[self.x <= 0] = 0.01
        self.x[self.x > 0] = 1
        return dloss * self.x

    def param(self):
        return []

class sequential(Module):

    def __init__(self, *args, eta):
        super().__init__()
        self.layers = args
        self.eta = eta

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
                weight, dw = par[0]
                #print("weight")
                #print(weight)
                #print(dw)
                weight -= dw * self.eta
                bias, db = par[1]
                #print("bias")
                #print(bias)
                #print(db)
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