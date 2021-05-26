import torch
import csv
import numpy as np
import time
from datetime import datetime

from six.moves import urllib

opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

from torch import optim
from torch import nn
from torch.nn import functional as F
from dlc_practical_prologue import *


def train_model(model, train_input, train_target, mini_batch_size, nb_epochs=25):
    criterion = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            output, _ = model(train_input.narrow(0, b, mini_batch_size))
            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))
            model.zero_grad()
            loss.backward()
            optimizer.step()


def train_model2(model, train_input, train_target, train_classes, mini_batch_size, nb_epochs=25):
    criterion = nn.MSELoss().to(device)
    criterion2 = nn.CrossEntropyLoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    for e in range(nb_epochs):
        for b in range(0, train_input.size(0), mini_batch_size):
            (output, output2) = model(train_input.narrow(0, b, mini_batch_size))
            loss = 0.8 * criterion(output, train_target.narrow(0, b, mini_batch_size)) + \
                   0.2 * criterion2(output2, train_classes.narrow(0, b, mini_batch_size).long())
            model.zero_grad()
            loss.backward()
            optimizer.step()


def compute_nb_errors(model, data_input, data_target, mini_batch_size):
    nb_data_errors = 0

    for b in range(0, data_input.size(0), mini_batch_size):
        output, _ = model(data_input.narrow(0, b, mini_batch_size))
        predicted_classes = (output > 0.5).float()
        for k in range(mini_batch_size):
            if data_target[b + k] != predicted_classes[k]:
                nb_data_errors = nb_data_errors + 1

    return nb_data_errors


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=5)
        self.conv3 = nn.Conv2d(24, 32, kernel_size=5)
        self.fc1 = nn.Linear(32 * 4 * 1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))

        x = F.relu(self.fc1(x.view(-1, 32 * 4 * 1)))
        x = F.relu(self.fc2(x))
        x = self.fc3(x).sum(1)
        return x, False


class Net_wh(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 24, kernel_size=3)
        self.conv2 = nn.Conv2d(24, 24, kernel_size=5)
        self.conv3 = nn.Conv2d(24, 24, kernel_size=5)
        self.fc1 = nn.Linear(24 * 4 * 2, 100)
        self.fc2 = nn.Linear(100, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)

    def forward(self, x):
        (x_1, x_2) = torch.split(x, 1, 1)
        x_1 = F.relu(self.conv1(x_1))
        x_2 = F.relu(self.conv1(x_2))
        x_1 = F.relu(self.conv2(x_1))
        x_2 = F.relu(self.conv2(x_2))
        x_1 = F.relu(F.max_pool2d(self.conv3(x_1), kernel_size=2))
        x_2 = F.relu(F.max_pool2d(self.conv3(x_2), kernel_size=2))

        x = torch.cat((x_1, x_2), 1)
        x = F.relu(self.fc1(x.view(-1, 24 * 4 * 2)))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.fc4(x).sum(1)
        return x, False


class Net_al(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(2, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 16, kernel_size=5)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=5)

        self.fc1 = nn.Linear(64 * 2 * 1, 100)
        self.fc2 = nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)
        self.fc1_cl = nn.Linear(64 * 2 * 1, 100)
        self.fc2_cl = nn.Linear(100, 50)
        self.fc3_cl = nn.Linear(50, 50)
        self.fc4_cl = nn.Linear(50, 20)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(F.max_pool2d(self.conv3(x), kernel_size=2))

        x_target = F.relu(self.fc1(x.view(-1, 64 * 2 * 1)))
        x_target = F.relu(self.fc2(x_target))
        x_target = self.fc3(x_target).sum(1)
        x_classes = F.relu(self.fc1_cl(x.view(-1, 64 * 2 * 1)))

        x_classes = F.relu(self.fc2_cl(x_classes))
        x_classes = F.relu(self.fc3_cl(x_classes))
        x_classes = F.relu(self.fc4_cl(x_classes))


        x_classes = x_classes.reshape([-1, 10, 2])

        return (x_target, x_classes)


class Net_wh_al(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5)

        self.fc1 = nn.Linear(32 * 4 * 2, 50)
        self.fc3 = nn.Linear(50, 20)
        self.fc4 = nn.Linear(20, 1)
        self.fc1_cl = nn.Linear(32 * 2 * 2, 100)
        self.fc2_cl = nn.Linear(100, 40)
        self.fc3_cl = nn.Linear(40, 10)

    def forward(self, x):
        (x_1, x_2) = torch.split(x, 1, 1)
        x_1 = F.relu(self.conv1(x_1))
        x_1 = F.relu(self.conv2(x_1))
        x_1 = F.relu(F.max_pool2d(self.conv3(x_1), kernel_size=2, stride=2))
        x_2 = F.relu(self.conv1(x_2))
        x_2 = F.relu(self.conv2(x_2))
        x_2 = F.relu(F.max_pool2d(self.conv3(x_2), kernel_size=2, stride=2))

        x = torch.cat((x_1, x_2), 1)
        x_target = F.relu(self.fc1(x.view(-1, 32 * 4 * 2)))
        x_target = F.relu(self.fc3(x_target))
        x_target = self.fc4(x_target).sum(1)
        x_1 = F.relu(self.fc1_cl(x_1.view(-1, 32 * 2 * 2)))
        x_1 = F.relu(self.fc2_cl(x_1))
        x_1 = F.relu(self.fc3_cl(x_1)).reshape([-1, 10, 1])
        x_2 = F.relu(self.fc1_cl(x_2.view(-1, 32 * 2 * 2)))
        x_2 = F.relu(self.fc2_cl(x_2))
        x_2 = F.relu(self.fc3_cl(x_2)).reshape([-1, 10, 1])

        x_classes = torch.cat((x_1, x_2), 2)
        return (x_target, x_classes)


mini_batch_size = 50
nb_epochs = 25
rangetest = 10
print_all = True

#############################
#Use of GPU
if torch.cuda.is_available():
    device = torch.device('cuda')
    print("Use of GPU")
else:
    device = torch.device('cpu')
    print("Use of CPU")
#############################
modelnow = Net_wh_al
if (modelnow == Net_al or modelnow == Net_wh_al):
    trainmethod = train_model2
    print("Use of AL")
else:
    trainmethod = train_model
    print("NO AL")
model = modelnow()
print(sum(p.numel() for p in model.parameters() if p.requires_grad))
time_start=datetime.now()
with open('data.csv', mode='w') as data_file:
    data_writer = csv.writer(data_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    data_writer.writerow(["Mean Train", "Var Train ", "Mean Test", "Var Test"])
    list_mean_train = []
    list_mean_test = []
    # print(ratio)
    for i in range(rangetest):
        model = modelnow().to(device)
        # print(sum(p.numel() for p in model.parameters() if p.requires_grad))
        train_input, train_target, train_classes, test_input, test_target, test_classes = generate_pair_sets(1000)

        #Move to gpu
        train_input = train_input.to(device)
        train_target = train_target.to(device)
        train_classes = train_classes.to(device)
        test_input = test_input.to(device)
        test_target = test_target.to(device)
        test_classes = test_classes.to(device)

        mean, std = train_input.mean(), train_input.std()
        train_input.sub_(mean).div_(std)
        test_input.sub_(mean).div_(std)

        #trainmethod(model, train_input, train_target.to(torch.float), mini_batch_size, nb_epochs)
        trainmethod(model, train_input, train_target.to(torch.float), train_classes.to(torch.float),
                        mini_batch_size, nb_epochs)

        train_error = compute_nb_errors(model, train_input, train_target, mini_batch_size) / train_input.size(
                0) * 100
        test_error = compute_nb_errors(model, test_input, test_target, mini_batch_size) / test_input.size(0) * 100
        if (print_all):
            print('        train_error {:.02f}% test_error {:.02f}%'.format(train_error, test_error))

        list_mean_train.append(train_error)
        list_mean_test.append(test_error)
        time_now = datetime.now()
        per_done = 100*(i+1)/rangetest
        est_time = (time_start+(time_now-time_start)*100/(per_done)).strftime("%H:%M:%S")

        print("Done: " + str(round(per_done,2))+ "/100" + " Heure de fin: " + str(est_time))


    val_mean_train = np.mean(list_mean_train)
    val_mean_test = np.mean(list_mean_test)
    val_var_train = np.var(list_mean_train)
    val_var_test = np.var(list_mean_test)
    mtr = str(round(val_mean_train, 2))
    vtr = str(round(val_var_train, 2))
    mte = str(round(val_mean_test, 2))
    vte = str(round(val_var_test, 2))

    print("Train error mean: " + mtr + " Train variance: " + vtr)
    print("Test error mean: " + mte + " Test variance: " + vte)

    data_writer.writerow({mtr, vtr, mte, vte})
