import math as math
import torch

def generate_data():
    train_input = torch.rand([1000, 2])
    test_input = torch.rand([1000, 2])
    train_label = get_label(train_input)
    test_label = get_label(test_input)
    return train_input, train_label, test_input, test_label

def get_label(input):
    return (torch.sqrt(torch.pow(input[:,1] - 0.5,2)) + torch.sqrt(torch.pow(input[:,0] - 0.5,2)) < 1/math.sqrt(2*math.pi)).float().view(-1,1)

def generate_data_lin():
    train_input = torch.rand([10, 2])
    test_input = torch.rand([10, 2])
    train_label = get_label_lin(train_input)
    #print(train_label.size())
    test_label = get_label_lin(test_input)
    return train_input, train_label, test_input, test_label

def get_label_lin(input):
    return (input[:,1] - input[:,0] < 0).float().view(-1,1)