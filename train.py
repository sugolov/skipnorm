import torch
from vit import ViT

device = "cuda" if torch.cuda.is_available() else "cpu"

# additional fn

if __name__ == "__main__":
    import os
    import argparse

    import torch.optim as optim
    from torch.utils.data import DataLoader

    import torchvision
    from torchvision import transforms

    from tqdm import tqdm
    import wandb

    # args

    parser = argparse.ArgumentParser("get args for training")
    parser.add_argument("--cifar", action="store_true")

    args = parser.parse_args()

    name = "ViT"

    if args.cifar:
        name += "_cifar"

        model_config = {
            "image_size": 32, 
            "patch_size": 4, 
            "num_classes": 10, 
            "dim": 64, 
            "depth": 10, 
            "heads": 8, 
            "mlp_dim": 256, 
            "channels": 3, 
            "dim_head": 64
        }

        model = ViT(**model_config).to(device)

        print(dir(ViT))

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = torchvision.datasets.CIFAR10(root="~/.datasets/", train=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root="~/.datasets/", train=False, transform=transform)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
        

    else: 
        pass


    wandb.init(project=name, config=model_config)
    
    optimizer = optim.Adam(model.parameters(), lr=2e-5)

    for epoch in range(1):
        
        for X, c in tqdm(train_loader):

            pred = model(X)
            loss = torch.nn.functional.cross_entropy(pred, c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({"loss": loss.item()})

    wandb.finish()







