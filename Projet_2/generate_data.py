import math as math
import torch

def generate_data():
    train_input = torch.rand([1000, 2])
    test_input = torch.rand([1000, 2])
    train_label = get_label(train_input)
    test_label = get_label(test_input)
    return train_input, train_label, test_input, test_label

def get_label(input):
    return (torch.sqrt(torch.pow(input[:,1] - 0.5,2)) + torch.sqrt(torch.pow(input[:,0] - 0.5,2)) < 1/math.sqrt(2*math.pi)).int()
