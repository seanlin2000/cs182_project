import torch
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F
from torch import nn


class Net(nn.Module):
    def __init__(self, input_size, num_classes):
        super(Net, self).__init__()
        self.layer1a = nn.Linear(input_size, input_size)
        self.layer1b = nn.BatchNorm1d(input_size)
        self.layer1c = nn.ReLU()
        self.layer1d = nn.Dropout()
        self.layer2a = nn.Linear(input_size, input_size)
        self.layer2b = nn.BatchNorm1d(input_size)
        self.layer2c = nn.ReLU()
        self.layer2d = nn.Dropout()
        self.layer3a = nn.Linear(input_size, input_size)
        self.layer3b = nn.BatchNorm1d(input_size)
        self.layer3c = nn.ReLU()
        self.layer3d = nn.Dropout()
        self.layer4 = nn.Linear(input_size, num_classes)


    def forward(self, x):
        # x = x.flatten(1)
        x = self.layer1a(x)
        x = self.layer1b(x)
        x = self.layer1c(x)
        x = self.layer1d(x)
        x = self.layer2a(x)
        x = self.layer2b(x)
        x = self.layer2c(x)
        x = self.layer2d(x)
        x = self.layer3a(x)
        x = self.layer3b(x)
        x = self.layer3c(x)
        x = self.layer3d(x)
        x = self.layer4(x)
        return x

def init_model(num_classes):
    resnet = torchvision.models.resnet101(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, num_ftrs)
    seanormanet = Net(num_ftrs, num_classes)
    model_conv = nn.Sequential(resnet, seanormanet)
    optimizer_conv = optim.Adam(seanormanet.parameters(), lr=0.001)
    return model_conv, optimizer_conv
