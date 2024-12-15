import os
import argparse
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision import transforms
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

def main():
    # Set up memory optimizations
    setup_memory_settings()
    
    parser = argparse.ArgumentParser("get args for training")
    parser.add_argument("--cifar", action="store_true")
    parser.add_argument("--mnist", action="store_true")
    parser.add_argument("--skipnorm", action="store_true")
    parser.add_argument("--data_dir", type=str, default="~/.datasets/")
    parser.add_argument("--weight_dir", type=str, default="~/.weights/")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    args = parser.parse_args()
    
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

if __name__ == "__main__":
    main()