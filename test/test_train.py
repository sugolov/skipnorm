import sys
sys.path.append("../traj-norm")
import torch

from skipnorm import SkipNorm
from vit import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

import os
import argparse

import torch.optim as optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from tqdm import tqdm
import wandb

def test_cifar_loaders():
    data_dir = "~/.datasets/"

    transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False,  download=True, transform=transform)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    print(len(test_data))

    return True

def test_eval():

    model_config = {
            "image_size": 32, 
            "patch_size": 4, 
            "num_classes": 10, 
            "dim": 32, 
            "depth": 1, 
            "heads": 2, 
            "mlp_dim": 10, 
            "channels": 3, 
            "dim_head": 10
        }
        
    model = ViT(**model_config).to(device)

    X = torch.randn((5,3,32,32))

    pred = model(X)

    print(pred)
    print(torch.max(pred, dim=-1))

def test_eval_sn_transformer():

    model_config = {
            "image_size": 32, 
            "patch_size": 4, 
            "num_classes": 10, 
            "dim": 32, 
            "depth": 4, 
            "heads": 2, 
            "mlp_dim": 10, 
            "channels": 3, 
            "dim_head": 10,
            "sn_window_size": 2
        }
        
    model = ViT(**model_config).to(device)

    X = torch.randn((5,3,32,32))

    pred = model(X)

    print(pred)
    print(torch.max(pred, dim=-1))

def test_eval_loop():

    data_dir = "~/.datasets/"

    transform = transforms.Compose([
            transforms.ToTensor()
        ])

    train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False,  download=True, transform=transform)
    n_test_data = len(test_data)

    train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    model_config = {
            "image_size": 32, 
            "patch_size": 4, 
            "num_classes": 10, 
            "dim": 32, 
            "depth": 1, 
            "heads": 2, 
            "mlp_dim": 10, 
            "channels": 3, 
            "dim_head": 10
        }
        
    model = ViT(**model_config).to(device)


    for X, c in tqdm(train_loader):
        correct = 0

        X, c = X.to(device), c.to(device)
        pred = model(X)
        c_pred = torch.max(pred, dim=-1).indices
        
        correct += torch.sum(c_pred == c)
    print(correct / n_test_data)

    


if __name__ == "__main__":
    with torch.no_grad():
        # test_cifar_loaders()
        # test_eval()
        # test_eval_loop()
        test_eval_sn_transformer()