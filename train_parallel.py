import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision import transforms
from tqdm import tqdm
import wandb
from vit import ViT

def setup(rank, world_size):
    """Initialize distributed training environment."""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up distributed training environment."""
    dist.destroy_process_group()

def get_model_config(args):
    """Get model configuration based on arguments."""
    model_config = {
        "image_size": 32,
        "patch_size": 2,
        "num_classes": 10,
        "dim": 256,
        "depth": 8,
        "heads": 8,
        "mlp_dim": 512,
        "channels": 3,
        "dim_head": 256
    }
    
    if args.skipnorm:
        model_config["sn_window"] = 4
    
    return model_config

def get_data_loaders(args, rank, world_size):
    """Create distributed data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    
    if args.cifar:
        train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, 
                                                 download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, 
                                                download=True, transform=transform)
    else:
        raise NotImplementedError("Only CIFAR-10 is implemented in this example")
    
    train_sampler = DistributedSampler(train_data, num_replicas=world_size, 
                                      rank=rank, shuffle=True)
    test_sampler = DistributedSampler(test_data, num_replicas=world_size, 
                                     rank=rank, shuffle=False)
    
    train_loader = DataLoader(train_data, batch_size=64, sampler=train_sampler, 
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_data, batch_size=64, sampler=test_sampler, 
                           num_workers=2, pin_memory=True)
    
    return train_loader, test_loader, train_sampler, len(test_data)

def eval_loop(model, test_loader, epoch, rank, n_test_data):
    """Evaluate model on test set."""
    model.eval()
    correct = torch.zeros(1).to(rank)
    
    with torch.no_grad():
        for X, c in test_loader:
            X, c = X.to(rank), c.to(rank)
            pred = model(X)
            c_pred = torch.max(pred, dim=1).indices
            correct += torch.sum(c_pred == c)
    
    # Gather results from all processes
    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        accuracy = (correct.item() / n_test_data)
        wandb.log({
            "eval_epoch": epoch,
            "eval_acc": accuracy
        })
    
    model.train()
    return correct.item() / n_test_data

def train(rank, world_size, args):
    """Main training function for each process."""
    setup(rank, world_size)
    
    # Get model configuration and name
    model_config = get_model_config(args)
    name = "ViT_SN" if args.skipnorm else "ViT"
    name += "_cifar" if args.cifar else ""
    
    # Save model config (only on rank 0)
    if rank == 0:
        torch.save(model_config, os.path.join(args.weight_dir, name + "_model_config"))
        wandb.init(project="skipnorm", name=name, config=model_config)
    
    # Create model and wrap with DDP
    model = ViT(**model_config).to(rank)
    model = DDP(model, device_ids=[rank])
    print("wrapped models")
    
    # Get data loaders
    train_loader, test_loader, train_sampler, n_test_data = get_data_loaders(args, rank, world_size)
    print("got loaders")
    
    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=2e-4, weight_decay=1e-5)
    
    # Initial evaluation
    if rank == 0:
        eval_loop(model, test_loader, 0, rank, n_test_data)
    
    # Training loop
    for epoch in range(300):
        train_sampler.set_epoch(epoch)  # Important for proper shuffling
        
        for X, c in train_loader:
            X, c = X.to(rank), c.to(rank)
            pred = model(X)
            loss = torch.nn.functional.cross_entropy(pred, c)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            if rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item()
                })
            print(f"epoch {epoch} loss {loss.item()}")
        
        # Evaluation
        if (epoch + 1) % 1 == 0:
            eval_loop(model, test_loader, epoch + 1, rank, n_test_data)
        
        # Checkpoint (only on rank 0)
        if rank == 0 and (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "model": model.module.state_dict()
                },
                os.path.join(args.weight_dir, name + f"-cpt-{epoch+1}")
            )
    
    if rank == 0:
        wandb.finish()
    
    cleanup()

def main():
    parser = argparse.ArgumentParser("get args for training")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--skipnorm", action="store_true")
    parser.add_argument("--data_dir", type=str, default="~/.datasets/")
    parser.add_argument("--weight_dir", type=str, default="~/.weights/")
    parser.add_argument("--world_size", type=int, default=4, 
                        help="Number of GPUs to use for training")
    args = parser.parse_args()
    
    # Expand user directory
    args.data_dir = os.path.expanduser(args.data_dir)
    args.weight_dir = os.path.expanduser(args.weight_dir)
    
    # Create weight directory if it doesn't exist
    os.makedirs(args.weight_dir, exist_ok=True)
    
    # Launch distributed training
    mp.spawn(train, args=(args.world_size, args), nprocs=args.world_size, join=True)

if __name__ == "__main__":
    main()