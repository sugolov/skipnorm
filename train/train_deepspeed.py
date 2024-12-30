import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb
import deepspeed
import json

def create_checkpoint(model, checkpoint_path, model_name, epoch, checkpoint_items):
    if len(checkpoint_path) == 0:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
    }
    checkpoint.update(checkpoint_items)

    save_name = model_name + f"_{epoch}.pt"
    save_path = os.path.join(checkpoint_path, save_name)
    model.save_checkpoint(checkpoint_path, save_name, client_state=checkpoint_items)

def main(model, train_dataloader, test_dataloader, epochs, 
         model_name, checkpoint_path, checkpoint_epoch, local_rank, 
         checkpoint_items=None, log_wandb=True):
    
    # Initialize DeepSpeed
    with open('ds_config.json') as f:
        ds_config = json.load(f)

    # Initialize DeepSpeed engine
    model_engine, optimizer, _, _ = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=train_dataloader.dataset
    )

    checkpoint_items = checkpoint_items if checkpoint_items else {}

    def eval(model_engine, test_dataloader, epoch):
        model_engine.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for X, c in tqdm(test_dataloader):
                X = X.to(model_engine.device)
                c = c.to(model_engine.device)
                
                pred = model_engine(X)
                c_pred = torch.max(pred, dim=1).indices
                
                correct += torch.sum(c_pred == c)
                total += len(c)
            
            # Gather metrics across all processes
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(correct)
                torch.distributed.all_reduce(torch.tensor(total, device=model_engine.device))
            
            acc = (correct / total).item()
            
            if log_wandb and local_rank == 0:
                wandb.log({
                    "eval_epoch": epoch,
                    "eval_acc": acc
                })
        return acc

    def train_step(model_engine, train_dataloader, epoch):
        model_engine.train()
        ce = nn.CrossEntropyLoss()
        
        for X, Y in tqdm(train_dataloader):
            X = X.to(model_engine.device)
            Y = Y.to(model_engine.device)

            outputs = model_engine(X)
            loss = ce(outputs, Y.long())

            model_engine.backward(loss)
            model_engine.step()

            if log_wandb and local_rank == 0:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": model_engine.get_lr()[0]
                })
    
    # main loop
    eval(model_engine, test_dataloader, 0)
    for epoch in tqdm(range(1, epochs+1)):
        train_step(model_engine, train_dataloader, epoch)
        acc = eval(model_engine, test_dataloader, epoch)

        if epoch % checkpoint_epoch == 0 and local_rank == 0:
            checkpoint_items.update({"accuracy": acc})
            create_checkpoint(model_engine, checkpoint_path, model_name, epoch, checkpoint_items)
            
    return model_engine, acc