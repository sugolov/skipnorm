import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from tqdm import tqdm
import wandb
from torch.cuda.amp import autocast, GradScaler

def setup_distributed():
    """Initialize distributed training setup"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    return local_rank, int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

def cleanup_distributed():
    """Clean up distributed training"""
    if dist.is_initialized():
        dist.destroy_process_group()

def create_distributed_checkpoint(model, checkpoint_path, model_name, epoch, checkpoint_items, rank):
    """Create checkpoints for distributed training"""
    if len(checkpoint_path) == 0 or rank != 0:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    }
    checkpoint.update(checkpoint_items)

    save_name = model_name + f"_{epoch}.pt"
    save_path = os.path.join(checkpoint_path, save_name)
    torch.save(checkpoint, save_path)

def distributed_eval(model, test_dataloader, epoch, device, rank, world_size, log_wandb=True):
    """Evaluation function for distributed training"""
    model.eval()
    correct = torch.zeros(1).to(device)
    total = torch.zeros(1).to(device)

    with torch.no_grad(), autocast(enabled=True):
        for X, c in test_dataloader:
            X, c = X.to(device), c.to(device)
            pred = model(X)
            c_pred = torch.max(pred, dim=1).indices
            correct += torch.sum(c_pred == c)
            total += c.size(0)

    dist.all_reduce(correct, op=dist.ReduceOp.SUM)
    dist.all_reduce(total, op=dist.ReduceOp.SUM)
    
    accuracy = (correct.item() / total.item())
    
    if rank == 0 and log_wandb:
        wandb.log({
            "eval_epoch": epoch,
            "eval_acc": accuracy
        })
    return accuracy

def main(model, optimizer, scheduler, train_dataloader, test_dataloader, epochs, 
         model_name, checkpoint_path, checkpoint_epoch, checkpoint_items=None, log_wandb=True,
         project_name="skipnorm"):
    """Main training loop for distributed training"""
    # Setup distributed training
    local_rank, rank, world_size = setup_distributed()
    device = torch.device(f"cuda:{local_rank}")
    
    # Move model to device and wrap in DDP
    model = model.to(device)
    model = DDP(model, device_ids=[local_rank])

    # Initialize gradient scaler for mixed precision
    scaler = GradScaler()
    checkpoint_items = checkpoint_items if checkpoint_items else {}

    # Loss function
    ce = nn.CrossEntropyLoss()

    # Initial evaluation
    distributed_eval(model, test_dataloader, 0, device, rank, world_size, log_wandb)

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        train_dataloader.sampler.set_epoch(epoch)
        
        progress_bar = tqdm(train_dataloader) if rank == 0 else train_dataloader
        
        for X, c in progress_bar:
            X, c = X.to(device), c.to(device)
            
            with autocast(enabled=True):
                pred = model(X)
                loss = ce(pred, c)
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if scheduler:
                scheduler.step()

            if rank == 0 and log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "learning_rate": scheduler.get_last_lr()[0] if scheduler else None
                })

        # Evaluation
        acc = distributed_eval(model, test_dataloader, epoch, device, rank, world_size, log_wandb)

        # Checkpointing
        if epoch % checkpoint_epoch == 0:
            checkpoint_items.update({"accuracy": acc})
            create_distributed_checkpoint(
                model, 
                checkpoint_path,
                model_name,
                epoch,
                checkpoint_items,
                rank
            )

    if rank == 0 and log_wandb:
        wandb.finish()
    
    cleanup_distributed()
    return model, acc