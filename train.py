import os
import argparse

import torch
from torch.utils.data import DataLoader

import torchvision
from torchvision import transforms

from vit import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

# additional fn

if __name__ == "__main__":
    # args

    parser = argparse.ArgumentParser("get args for training")
    parser.add_argument("--cifar", action="store_true")

    args = parser.parse_args()

    if args.cifar:
        print("yo")
        model = ViT(
            image_size=32, 
            patch_size=4, 
            num_classes=10, 
            dim=64, 
            depth=10, 
            heads=8, 
            mlp_dim=256, 
            channels = 3, 
            dim_head = 64
        ).to(device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = torchvision.datasets.CIFAR10(root="~/.datasets/", train=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root="~/.datasets/", train=False, transform=transform)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)

    else: 
        pass

    for epoch in range(1):
        
        for X, c in train_loader:
            pred = model(X)
            print(torch.nn.functional.cross_entropy(pred, c))
        







