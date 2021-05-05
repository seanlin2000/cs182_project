"""
This file will train a sample network on the tiny image-net data. It should be
your final goal to improve on the performance of this model by swapping out large
portions of the code. We provide this model in order to test the full pipeline,
and to validate your own code submission.
"""

from __future__ import print_function, division

# import sys
# sys.path.append(root_folder)
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
import pathlib
# from utils import dictionary, preprocess_val

from models.convengers import Thor, OGNet, init_model
from models.solver import NickFury
import model

def main():
    dataset_folder = "./data/tiny-imagenet-200/"
    data_dir = pathlib.Path(dataset_folder)

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.Resize(256),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the data
    image_datasets = {x: datasets.ImageFolder(data_dir / x, data_transforms[x]) for x in ['train', 'val']}
    # Set num_workers=2 when we use CPU, 4 when we use GPU, batch size needs to be smaller for weaker CPUs
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50, shuffle=True, num_workers=4,pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    '''
    resnet = torchvision.models.resnet101(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    num_ftrs = resnet.fc.in_features
    resnet.fc = nn.Linear(num_ftrs, 200)
    '''
    
    resnet = Thor(num_blocks=1, requires_grad=False)
    print(device)
    
    resnet = resnet.to(device)
    model_solver = NickFury(resnet, dataloaders, dataset_sizes)
    
    model_criterion = nn.CrossEntropyLoss()
    model_optimizer = optim.Adam(resnet.parameters(), lr=0.001)
    model_exp_lr_scheduler = lr_scheduler.StepLR(model_optimizer, step_size=7, gamma=0.1)

    
    model_loss_history = model_solver.train(model_optimizer, model_criterion, model_exp_lr_scheduler, device)
    
    return model_test


if __name__ == '__main__':
    main()
