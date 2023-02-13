import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import random_split
import torchvision

# Loads the dataset into a single general dataloader
def imageNet_dataloader(root_dir, batch_size, num_workers):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = torchvision.datasets.ImageFolder(root=root_dir, transform=transform)

    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    print('Dataset loaded!')

    return loader

# Loads the dataset into a training/validation split dataloaders, with ratio as a parameter.
def imageNet_dataloader_tv(root_dir, batch_size, num_workers, ratio=0.8):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    dataset = torchvision.datasets.ImageFolder(root='./data/ILSVRC2012_img_train', transform=transform)

    train_size = int(ratio * len(dataset))  
    val_size = len(dataset) - train_size  

    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    return train_loader, val_loader

def cifar100_dataloader(root_dir, batch_size, num_workers, download=True):
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    train_set = torchvision.datasets.CIFAR100(root=root_dir, train=True,
                                            download=download, transform=transform)

    test_set = torchvision.datasets.CIFAR100(root=root_dir, train=False,
                                           download=download, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                              shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                             shuffle=False, num_workers=num_workers)

    return train_loader, test_loader
