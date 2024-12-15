import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from vit import ViT
import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
import torchvision
from torchvision.transforms import transforms
from torch.utils.data.distributed import DistributedSampler
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm
import wandb
import gc
from torch.cuda.amp import autocast, GradScaler
from vit import ViT

def setup_memory_settings():
    """Configure CUDA memory settings to avoid fragmentation."""
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

def get_data_loaders(args, world_size=None, rank=None):
    """Create memory-efficient data loaders."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.half())  # Convert to half precision
    ])
    
    if args.cifar:
        train_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, 
                                                 download=True, transform=transform)
        test_data = torchvision.datasets.CIFAR10(root=args.data_dir, train=False, 
                                                download=True, transform=transform)
    
    if world_size is not None and rank is not None:
        train_sampler = DistributedSampler(train_data, num_replicas=world_size, 
                                          rank=rank, shuffle=True)
        test_sampler = DistributedSampler(test_data, num_replicas=world_size, 
                                         rank=rank, shuffle=False)
    else:
        train_sampler = None
        test_sampler = None
    
    # Reduce batch size and increase num_workers
    train_loader = DataLoader(
        train_data, 
        batch_size=32,  # Reduced from 64
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    test_loader = DataLoader(
        test_data, 
        batch_size=32,  # Reduced from 64
        shuffle=False,
        sampler=test_sampler,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2
    )
    
    return train_loader, test_loader, train_sampler, len(test_data)
=======

from collections import defaultdict

from cosine_warmup import CosineWarmupScheduler

"""
ViT_config = {
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
"""

ViT_config = {
    "image_size": 32, 
    "patch_size": 2, 
    "num_classes": 10, 
    "dim": 768, 
    "depth": 12, 
    "heads": 12, 
    "mlp_dim": 3072, 
    "channels": 3, 
    "dim_head": 768
}

train_config = {
    "lr": 3e-3,
    "eta_min": 1e-5,
    "weight_decay": 1e-3,
    "batch_size": 16,
    "epochs": 300
}

def log_gpu_memory(message="", rank=0):
    if rank == 0:  # Only print for rank 0
        allocated = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
        cached = torch.cuda.memory_reserved() / (1024 * 1024)  # Convert to MB
        print(f"[{message}] GPU Memory: {allocated:.2f}MB allocated, {cached:.2f}MB cached")

def gather_losses(local_loss, rank, world_size):
    """Gather losses from all GPUs."""
    losses = [torch.zeros_like(local_loss) for _ in range(world_size)]
    dist.all_gather(losses, local_loss)
    return {i: loss.item() for i, loss in enumerate(losses)}

def setup_distributed():
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))

def cleanup_distributed():
    dist.destroy_process_group()

def eval_loop(model, test_loader, epoch, device, rank):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad(), autocast(enabled=True):  # Enable AMP for eval
        for X, c in test_loader:
            X, c = X.to(device), c.to(device)
            pred = model(X)
            c_pred = torch.max(pred, dim=1).indices
            correct += torch.sum(c_pred == c)
            total += c.size(0)

    correct_tensor = torch.tensor([correct], device=device)
    total_tensor = torch.tensor([total], device=device)
    
    dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
    dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)
    
    if rank == 0:
        accuracy = (correct_tensor.item() / total_tensor.item())
        wandb.log({
            "eval_epoch": epoch,
            "eval_acc": accuracy
        })
>>>>>>> f482ade37791e42d93af67735984455c7c31a8c7

def main():
    # Set up memory optimizations
    setup_memory_settings()
    
    parser = argparse.ArgumentParser("get args for training")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--skipnorm", action="store_true")
    parser.add_argument("--data_dir", type=str, default="~/.datasets/")
    parser.add_argument("--weight_dir", type=str, default="~/.weights/")
<<<<<<< HEAD
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
=======
>>>>>>> f482ade37791e42d93af67735984455c7c31a8c7
    args = parser.parse_args()

    # Setup distributed training
    setup_distributed()
    local_rank = int(os.environ["LOCAL_RANK"])
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    device = torch.device(f"cuda:{local_rank}")

    #hypers
    epochs = train_config["epochs"]

    # Initialize gradient scaler for AMP
    scaler = GradScaler()

    # Model setup
    name = ""
    if args.skipnorm:
        name += "ViT_SN"
        model_config = ViT_config
        model_config["sn_window"] = 6
        if rank == 0:
            torch.save(model_config, os.path.join(args.weight_dir, name + "_model_config"))
        model = ViT(**model_config).to(device)
    else:
        name += "ViT"
        model_config = ViT_config
        if rank == 0:
            torch.save(model_config, os.path.join(args.weight_dir, name + "_model_config"))
        model = ViT(**model_config).to(device)

    model = DDP(model, device_ids=[local_rank])

    # Data loading
    if args.cifar:
        name += "_cifar"
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        train_data = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=True, download=rank == 0, transform=transform
        )
        test_data = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=rank == 0, transform=transform
        )

        train_sampler = DistributedSampler(
            train_data, 
            num_replicas=world_size,
            rank=rank
        )
        test_sampler = DistributedSampler(
            test_data,
            num_replicas=world_size,
            rank=rank
        )

        train_loader = DataLoader(
            train_data, 
            batch_size=train_config["batch_size"], 
            sampler=train_sampler,
            num_workers=2,
            pin_memory=True
        )
        test_loader = DataLoader(
            test_data,
            batch_size=train_config["batch_size"],
            sampler=test_sampler,
            num_workers=2,
            pin_memory=True
        )

    # Initialize optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=train_config["lr"], 
        weight_decay=train_config["weight_decay"]
        )
    #scheduler = CosineAnnealingLR(
    #    optimizer, 
    #    T_max=epochs * int(), 
    #    eta_min=train_config["eta_min"]
    #)
    steps_per_epoch = int( len(train_data) / (world_size * train_config["batch_size"]) )
    print(steps_per_epoch)
    scheduler = CosineWarmupScheduler(
        optimizer,
        warmup_epochs=3 * steps_per_epoch,
        max_epochs=100 * steps_per_epoch,
        warmup_start_lr=1e-5,
        eta_min=1e-5
    )

    log_gpu_memory()

    # Initialize wandb only on rank 0
    
<<<<<<< HEAD
    # Setup distributed training
    if 'LOCAL_RANK' in os.environ:
        local_rank = int(os.environ['LOCAL_RANK'])
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend='nccl')
    else:
        local_rank = 0
        rank = None
        world_size = None
    
    device = torch.device(f'cuda:{local_rank}')
    
    # Clear GPU cache
    torch.cuda.empty_cache()
    gc.collect()
    
    # Model configuration with reduced size
    model_config = {
        "image_size": 32,
        "patch_size": 2,
        "num_classes": 10,
        "dim": 192,       # Reduced from 256
        "depth": 6,       # Reduced from 8
        "heads": 6,       # Reduced from 8
        "mlp_dim": 384,   # Reduced from 512
        "channels": 3,
        "dim_head": 192   # Reduced from 256
    }
    
    if args.skipnorm:
        name = "ViT_SN"
        model_config["sn_window"] = 4
    else:
        name = "ViT"
    
    if args.cifar:
        name += "_cifar"
    
    # Initialize model with mixed precision
    model = ViT(**model_config).to(device).half()  # Convert to half precision
    
    if rank is not None:
        model = DDP(model, device_ids=[local_rank])
    
    # Initialize mixed precision training
    scaler = GradScaler()
    
    # Get data loaders
    train_loader, test_loader, train_sampler, n_test_data = get_data_loaders(
        args, world_size, rank
    )
    
    # Initialize optimizer with gradient clipping
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=2e-4,
        weight_decay=1e-5,
        eps=1e-4  # Increased epsilon for half precision
    )
    
    # Training loop with memory optimizations
    for epoch in range(300):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        
        model.train()
        optimizer.zero_grad(set_to_none=True)  # More memory efficient than zeros
        
        for i, (X, c) in enumerate(tqdm(train_loader, disable=rank not in [0, None])):
            X, c = X.to(device, non_blocking=True), c.to(device, non_blocking=True)
            
            # Mixed precision training
            with autocast():
                pred = model(X)
                loss = torch.nn.functional.cross_entropy(pred, c)
                loss = loss / args.gradient_accumulation_steps
            
            # Scale loss and backward pass
            scaler.scale(loss).backward()
            
            # Gradient accumulation
            if (i + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            
            if rank in [0, None]:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item() * args.gradient_accumulation_steps
                })
            
            # Clear cache periodically
            if i % 100 == 0:
                torch.cuda.empty_cache()
        
        # Checkpoint
        if rank in [0, None] and (epoch + 1) % 10 == 0:
            state_dict = model.module.state_dict() if rank is not None else model.state_dict()
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "model": state_dict,
                    "scaler": scaler.state_dict(),
                },
                os.path.join(args.weight_dir, name + f"-cpt-{epoch+1}")
            )
    
    if rank is not None:
        dist.destroy_process_group()
=======
    if rank == 0:
        param_log = {}
        [param_log.update(_) for _ in [ViT_config, train_config]]
        wandb.init(project="skipnorm", name=f"{name}_distributed_amp", config=param_log)

    running_losses = defaultdict(float)
    step_count = 0

    # Pre-training eval
    eval_loop(model, test_loader, 0, device, rank)

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_sampler.set_epoch(epoch)
        
        if rank == 0:
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}")
        else:
            progress_bar = train_loader

        for X, c in progress_bar:
            X, c = X.to(device), c.to(device)
            
            # Use autocast for mixed precision training
            with autocast(enabled=True):
                pred = model(X)
                loss = torch.nn.functional.cross_entropy(pred, c)
            
            optimizer.zero_grad()
            # Use the scaler for backwards pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            scheduler.step()

            # Gather and log per-GPU losses
            gpu_losses = gather_losses(loss.detach(), rank, world_size)
            for gpu_rank, gpu_loss in gpu_losses.items():
                running_losses[gpu_rank] += gpu_loss
            step_count += 1

            # Log both aggregated and per-GPU losses
            if rank == 0:
                # Log overall loss
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0]
                })
                
                # Log individual GPU losses
                wandb.log({
                    f"loss_gpu_{gpu_rank}": gpu_loss 
                    for gpu_rank, gpu_loss in gpu_losses.items()
                })

                # Update progress bar with all GPU losses
                progress_bar.set_postfix({
                    f"GPU{gpu_rank}_loss": f"{gpu_loss:.4f}"
                    for gpu_rank, gpu_loss in gpu_losses.items()
                })

            #if rank == 0:
            #    wandb.log({
            #        "epoch": epoch,
            #        "loss": loss.item(),
            #        "learning_rate": scheduler.get_last_lr()[0]
            #    })
        
        log_gpu_memory(f"fp epoch {epoch}")

        # Evaluation
        if (epoch + 1) % 1 == 0:
            eval_loop(model, test_loader, epoch+1, device, rank)

        # Log average losses for the epoch
        if rank == 0:
            avg_losses = {
                f"avg_loss_gpu_{gpu_rank}": total_loss / step_count
                for gpu_rank, total_loss in running_losses.items()
            }
            wandb.log({
                "epoch": epoch,
                **avg_losses
            })

        # Checkpointing (only on rank 0)
        if rank == 0 and (epoch + 1) % 10 == 0:
            torch.save(
                {
                    "optimizer": optimizer.state_dict(),
                    "model": model.module.state_dict(),
                    "scaler": scaler.state_dict(),  # Save scaler state
                    "scheduler": scheduler.state_dict() 
                },
                os.path.join(args.weight_dir, name + f"-cpt-{epoch+1}")
            )

    if rank == 0:
        wandb.finish()
    
    cleanup_distributed()
>>>>>>> f482ade37791e42d93af67735984455c7c31a8c7

if __name__ == "__main__":
    main()
