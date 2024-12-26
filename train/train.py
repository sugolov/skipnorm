import torch
from vit import ViT
import wandb

def main(model, optimizer, scheduler, train_dataloader, test_dataloader, epochs, log_wandb=True):

    steps_per_epoch = int(len(train_loader) / batch_size)  

    ce = nn.CrossEntropyLoss()

    def eval(model, test_loader, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, c in tqdm(test_loader):
                X, c = X.to(device), c.to(device)
                pred = model(X)
                c_pred = torch.max(pred, dim=1).indices
                
                correct += torch.sum(c_pred == c)
                total += len(c)
            acc = (correct / total).item()
            wandb.log({
                "eval_epoch": epoch,
                "eval_acc": acc
            })

        return acc

    def train_step(model, train_loader, epoch):
        for X, Y in train_loader:

            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            Y_pred =  model(X)
            loss = ce(Y_pred, Y.long())

            loss.backward()
            optimizer.step()
            scheduler.step() if scheduler else None

            wandb.log({"epoch": epoch, "loss": loss.item(), "lr": scheduler.get_lr()[0] if scheduler else None}) if log_wandb else None

    # training loop

    eval(model, test_loader, 0)
    for epoch in tqdm(range(1, epochs+1)):
        train_step(model, train_loader, epoch)
        acc = eval(model, test_loader, epoch)
        # TODO: add checkpointing
            
    return model, acc


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

    name = ""

    if args.skipnorm:
        name += "ViT_SN"

        model_config = ViT_config
        model_config["sn_window"] = 6

        torch.save(model_config, os.path.join(args.weight_dir, name + f"_model_config"))
        model = ViT(**model_config).to(device)
    else:
        name += "ViT"

        model_config = ViT_config

        torch.save(model_config, os.path.join(args.weight_dir, name + f"_model_config"))
        model = ViT(**model_config).to(device)

    if args.cifar:
        name += "_cifar"

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

    # optimizer 
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)

    # train loop
    wandb.init(project="skipnorm", name=name, config=model_config)

    # define eval loop
    def eval_loop(model, test_loader, epoch):
        correct = 0
        with torch.no_grad():
            for X, c in tqdm(test_loader):

                X, c = X.to(device), c.to(device)
                pred = model(X)
                c_pred = torch.max(pred, dim=1).indices
                
                correct += torch.sum(c_pred == c)

            wandb.log({
                "eval_epoch": epoch,
                "eval_acc": (correct / n_test_data).item()
            })

    eval_loop(model, test_loader, 0)

    for epoch in range(300):
        
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
        if (epoch + 1) % 1 == 0:
            eval_loop(model, test_loader, epoch+1)

        # checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(
                {"optimizer": optimizer.state_dict(),
                "model": model.state_dict()},
                os.path.join(args.weight_dir, name + f"-cpt-{epoch+1}")
            ) 

    wandb.finish()