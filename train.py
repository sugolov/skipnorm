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
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--skipnorm", action="store_true")
    parser.add_argument("--data_dir", type=str, default="~/.datasets/")
    parser.add_argument("--weight_dir", type=str, default="~/.weights/")
    args = parser.parse_args()

    data_dir = args.data_dir
    weight_dir = args.data_dir

    name = "ViT"

    if args.cifar:
        name += "_cifar"

        model_config = {
            "image_size": 32, 
            "patch_size": 4, 
            "num_classes": 10, 
            "dim": 256, 
            "depth": 10, 
            "heads": 8, 
            "mlp_dim": 256, 
            "channels": 3, 
            "dim_head": 128
        }

        torch.save(model_config, os.path.join(args.weight_dir, name + f"_model_config"))

        model = ViT(**model_config).to(device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_data = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root=data_dir, train=False,  download=True, transform=transform)
        n_test_data = len(test_data)

        train_loader = DataLoader(train_data, batch_size=64, shuffle=False, num_workers=2)
        test_loader = DataLoader(test_data, batch_size=64, shuffle=False, num_workers=2)
        

    else: 
        pass

    # train loop
    wandb.init(project=name, config=model_config)

    optimizer = optim.Adam(model.parameters(), lr=2e-4)
    for epoch in range(200):
        
        for X, c in tqdm(train_loader):
            X, c = X.to(device), c.to(device)

            pred = model(X)
            loss = torch.nn.functional.cross_entropy(pred, c)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            wandb.log({
                "epoch": epoch,
                "loss": loss.item()
            })   

        # eval loop
        if (epoch + 1) % 10 == 0:
            with torch.no_grad():
                for X, c in tqdm(train_loader):
                    correct = 0

                    X, c = X.to(device), c.to(device)
                    pred = model(X)
                    c_pred = torch.max(pred, dim=-1).indices
                    
                    correct += torch.sum(c_pred == c)
            accuracy = correct / n_test_data

            wandb.log({
                "eval_epoch": epoch,
                "eval_acc": accuracy.item()
            })
            

        # checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {"optimizer": optimizer.state_dict(),
                "model": model.state_dict()},
                os.path.join(args.weight_dir, name + f"-cpt-{epoch}")
            ) 

    wandb.finish()







