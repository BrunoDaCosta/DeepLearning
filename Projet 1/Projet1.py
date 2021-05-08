import torch
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

from torch import optim
from torch import nn
from torch.nn import functional as F
from dlc_practical_prologue import *


def train_model(model, train_input, train_target, mini_batch_size, nb_epochs=25):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output,_ = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def train_model2(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=25):
    criterion = nn.MSELoss()
    optimizer = optim.SGD(model.parameters(), lr = 1e-1)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            (output, output2) = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size)) + 0.3*criterion(output2, train_classes.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()

def compute_nb_errors(model, data_input, data_target, mini_batch_size):

    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output,_ = model(data_input.narrow(0, b, mini_batch_size))
        predicted_classes = (output>0.5).float()
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors

class Net(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)
        self.fc1 = nn.Linear(32*4*1, 50)
        self.fc2 = nn.Linear(50, 20)
        self.fc3 = nn.Linear(20, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2, stride=2))
        
        x = F.relu(self.fc1(x.view(-1,32*4*1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).sum(1)
        return x, False

class Net_wh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=5)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5)
        self.fc1 = nn.Linear(24*4*2, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        (x_1,x_2) = torch.split(x, 1, 1)
        x_1 = F.relu(self.conv1(x_1))
        x_2 = F.relu(self.conv1(x_2))
        x_1 = F.relu(self.conv2(x_1))
        x_2 = F.relu(self.conv2(x_2))
        x_1 = F.relu(F.max_pool2d(self.conv3(x_1), kernel_size=2))
        x_2 = F.relu(F.max_pool2d(self.conv3(x_2), kernel_size=2))
        
        x = torch.cat((x_1, x_2),1)
        x = F.relu(self.fc1(x.view(-1,24*4*2)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).sum(1)
        return x, False


class Net_al(nn.Module):
    def _init_(self):
        super()._init_()
        self.conv1 = nn.Conv2d(2, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)

        self.fc1 = nn.Linear(64*2*1, 100)
        self.fc2 = nn.Linear(100, 1)
        self.fc1_cl = nn.Linear(64*2*1, 100)
        self.fc2_cl = nn.Linear(100, 10)
        self.fc3_cl = nn.Linear(100, 10)
        
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))
        
        x_target = F.relu(self.fc1(x.view(-1,64*2*1)))
        x_target = self.fc2(x_target).sum(1)
        x_classes = F.relu(self.fc1_cl(x.view(-1, 64*2*1)))

        x_classes_1 = self.fc2_cl(x_classes)
        x_classes_2 = self.fc3_cl(x_classes)

        x_classes_1 = torch.argmax(x_classes_1,1)
        x_classes_2 = torch.argmax(x_classes_2,1)

        x_classes = torch.cat((x_classes_1.view(-1,1), x_classes_2.view(-1,1)), 1)


        return (x_target, x_classes)

    
class Net_wh_al(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        
        self.fc1 = nn.Linear(64*2*2, 100)
        self.fc2 = nn.Linear(100, 1)
        self.fc1_cl = nn.Linear(64*2*2, 100)
        self.fc2_cl = nn.Linear(100, 2)

    def forward(self, x):
        (x_1,x_2) = torch.split(x, 1, 1)
        x_1 = F.relu(F.max_pool2d(self.conv1(x_1), kernel_size=2))
        x_2 = F.relu(F.max_pool2d(self.conv1(x_2), kernel_size=2))
        x_1 = F.relu(F.max_pool2d(self.conv2(x_1), kernel_size=2))
        x_2 = F.relu(F.max_pool2d(self.conv2(x_2), kernel_size=2))
        
        x = torch.cat((x_1, x_2),1)
        
        x_target = F.relu(self.fc1(x.view(-1,64*2*2)))
        x_target = self.fc2(x_target).sum(1)
        x_classes = F.relu(self.fc1_cl(x.view(-1, 64*2*2)))
        x_classes = self.fc2_cl(x_classes)

        return (x_target, x_classes)

mini_batch_size = 50
nb_epochs = 25

repeats = 10
err=torch.empty(2,repeats)
for i in range(repeats):
    model = Net_wh()
    if i==0:
        print("Number of params: {}".format(sum(p.numel() for p in model.parameters() if p.requires_grad)))
    
    train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)
    mean, std = train_input.mean(), train_input.std()
    train_input.sub_(mean).div_(std)
    test_input.sub_(mean).div_(std)

    train_model(model, train_input, train_target.to(torch.float), mini_batch_size, nb_epochs)                                                                                                    
    ##train_model2(model, train_input, train_target.to(torch.float), train_classes.to(torch.float), mini_batch_size, nb_epochs)

    err[0][i] = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(0) * 100
    err[1][i] = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100 
    print('{}/{} - train_error {:.02f}% test_error {:.02f}%'.format(i+1, repeats, err[0][i], err[1][i]))
print("Mean training error: {:.02f}%".format(err[1].mean()))

import matplotlib.pyplot as plt
plt.boxplot(err[1])
