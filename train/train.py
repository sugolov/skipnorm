import os
import torch
import torch.nn as nn
from tqdm import tqdm
import wandb

def create_checkpoint(model, checkpoint_path, model_name, epoch, checkpoint_items):
    if len(checkpoint_path) == 0:
        return

    os.makedirs(checkpoint_path, exist_ok=True)
    checkpoint = {
        "epoch": epoch, 
        "model": model.state_dict()
    }
    checkpoint.update(checkpoint_items)

    save_name = model_name + f"_{epoch}.pt"
    save_path = os.path.join(checkpoint_path, save_name)
    torch.save(checkpoint, save_path)
     

def main(model, optimizer, scheduler, train_dataloader, test_dataloader, epochs, 
    model_name, checkpoint_path, checkpoint_epoch, checkpoint_items=None, log_wandb=True):
    
    device = next(model.parameters()).device  # Get device from model
    batch_size = train_dataloader.batch_size
    steps_per_epoch = len(train_dataloader)
    checkpoint_items = checkpoint_items if checkpoint_items else {}

    def eval(model, test_dataloader, epoch):
        correct = 0
        total = 0
        with torch.no_grad():
            for X, c in tqdm(test_dataloader):
                X, c = X.to(device), c.to(device)
                pred = model(X)
                c_pred = torch.max(pred, dim=1).indices
                
                correct += torch.sum(c_pred == c)
                total += len(c)
            acc = (correct / total).item()
            if log_wandb:
                wandb.log({
                    "eval_epoch": epoch,
                    "eval_acc": acc
                })
        return acc

    def train_step(model, train_dataloader, epoch):
        for X, Y in tqdm(train_dataloader):
            X, Y = X.to(device), Y.to(device)

            optimizer.zero_grad()

            Y_pred = model(X)
            loss = ce(Y_pred, Y.long())

            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()

            if log_wandb:
                wandb.log({
                    "epoch": epoch,
                    "loss": loss.item(),
                    "lr": scheduler.get_lr()[0] if scheduler else None
                })

    ce = nn.CrossEntropyLoss()
    
    # main loop
    eval(model, test_dataloader, 0)
    for epoch in tqdm(range(1, epochs+1)):
        train_step(model, train_dataloader, epoch)
        acc = eval(model, test_dataloader, epoch)


        if epoch % checkpoint_epoch == 0:
            checkpoint_items.update({"accuracy": acc})
            create_checkpoint(model, checkpoint_path, model_name, epoch, checkpoint_items)
            
    return model, acc