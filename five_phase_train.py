import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms

import pathlib
from models.convengers import *
from models.solver import NickFury

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
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=5)], p=0.25),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    # Load the data
    image_datasets = {x: datasets.ImageFolder(data_dir / x, data_transforms[x]) for x in ['train', 'val']}
    # Set num_workers=2 when we use CPU, 4 when we use GPU, batch size needs to be smaller for weaker CPUs
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=50, shuffle=True, num_workers=2,pin_memory=True) for x in ['train', 'val']}
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    concat_conv = ConvengersCat(requires_grad=False)
    concat_conv = concat_conv.to(device)
    
    state_dict = torch.load("three_phase_finished.pt")["net"]
    concat_conv.load_state_dict(state)
    
    concat_solver = NickFury("concat_conv", concat_conv, dataloaders, dataset_sizes, device)
    concat_criterion = nn.CrossEntropyLoss()
    
    fc_optimizer = optim.Adam(concat_conv.parameters(), lr=0.0001) # fc optimizer
    thor_optimizer = optim.Adam(concat_conv.parameters(), lr=0.00001) # ResNet optimizer
    ironman_optimizer = optim.Adam(concat_conv.parameters(), lr=0.00005) # Inception optimizer
    captainamerica_optimizer = optim.Adam(concat_conv.parameters(), lr=0.0001) # VGG optimizer
    concat_optimizer = optim.Adam(concat_conv.parameters(), lr=0.00001) # end-to-end optimizer

    thor_lr_scheduler = lr_scheduler.StepLR(thor_optimizer, step_size=5, gamma=0.1) # end-to-end lr-scheduler
    ironman_lr_scheduler = lr_scheduler.StepLR(ironman_optimizer, step_size=5, gamma=0.1) # end-to-end lr-scheduler
    captainamerica_lr_scheduler = lr_scheduler.StepLR(captainamerica_optimizer, step_size=5, gamma=0.1) # end-to-end lr-scheduler

    concat_lr_scheduler = lr_scheduler.StepLR(concat_optimizer, step_size=7, gamma=0.1) # end-to-end lr-scheduler
    
    # 5 phase train
    
    # First: train new FC layers
    
    '''
    print("Training SeaNorman Layers")
    concat_solver.train(fc_optimizer, concat_criterion, num_epochs=5)

    for i in range(1):
        # Second: train FC + Thor
        print("Training SeaNorman + Thor")
        concat_conv.thor_grad(True)
        concat_solver.train(thor_optimizer, concat_criterion, num_epochs=2, adv_train=True)
        concat_conv.thor_grad(False)

        # Third: train FC + IronMan
        print("Training SeaNorman + Iron Man")
        concat_conv.ironman_grad(True)
        concat_solver.train(ironman_optimizer, concat_criterion, num_epochs=2, adv_train=True)
        concat_conv.ironman_grad(False)
    '''
    '''
        # Fourth: train FC + CaptainAmerica
    print("Training SeaNorman + Captain America")
    concat_conv.captainamerica_grad(True)
    concat_solver.train(captainamerica_optimizer, concat_criterion, num_epochs=2, adv_train=True)
    concat_conv.captainamerica_grad(False)
    '''
    
    # Fifth: End to End optimizer
    print("Training end to end")
    concat_conv.end_to_end_grad(True)
    concat_solver.train(concat_optimizer, concat_criterion, concat_lr_scheduler, num_epochs=15, adv_train=True)
    
    concat_solver.save_model("five-phase-train-final", "concat.pt")
    #concat_solver.save_loss_history("concat_loss_history.pt")
    return model_test

main()