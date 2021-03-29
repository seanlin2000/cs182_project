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
from utils import dictionary
import model

def basic_train(model, optim, criterion, train_loader, num_epochs):
    start_time = time.time()
    for i in range(num_epochs):
        epoch_start_time = time.time()
        train_total, train_correct = 0,0
        for idx, (inputs, targets) in enumerate(train_loader):
        #   with torch.cuda.device(torch.cuda.current_device()):
        #     torch.cuda.empty_cache()
            optim.zero_grad()
        #   inputs = inputs.to(device)
        #   targets = targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optim.step()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            print("\r", end='')
            print(f'training {100 * idx / len(train_loader):.2f}%: {train_correct / train_total:.3f}', end='')
        epoch_end_time = time.time()
        hours, rem = divmod(epoch_end_time-epoch_start_time, 3600)
        minutes, seconds = divmod(rem, 60)
        print()
        print("Epoch {} completed with overall accuracy at {:.4f}".format(i, train_correct / train_total))
        print("Epoch {} completed with elapsed time {:0>2}:{:0>2}:{:05.2f}".format(i, int(hours),int(minutes),seconds))
        torch.save({
            'net': model.state_dict(),
        }, 'latest.pt')
    end_time = time.time()
    hours, rem = divmod(end_time-start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Training completed with total elapsed time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))
    

def main():
    dataset_folder = "../tiny-imagenet-200/"
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
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=20, shuffle=True, num_workers=2) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define the model
    model_conv, optimizer_conv = model.init_model(len(class_names))

    criterion = nn.CrossEntropyLoss()

    # Decay LR by a factor of 0.1 every 7 epochs
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

    # with torch.cuda.device(torch.cuda.current_device()):
    #     torch.cuda.empty_cache()
    model_conv = basic_train(model_conv, optimizer_conv, criterion, dataloaders['train'], num_epochs=25)

if __name__ == '__main__':
    main()