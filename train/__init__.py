import math
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim
    
def get_optimizer(opt_name, model, lr, momentum=0.9, weight_decay=0.05, opt_eps=1e-8, opt_betas=None):

    if opt_betas is None:
        opt_betas = (0.9, 0.999)
    
    opt_name = opt_name.lower()
    
    if opt_name == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif opt_name == 'adamw':
        optimizer = optim.AdamW(model.parameters(), lr=lr, betas=opt_betas, eps=opt_eps, weight_decay=weight_decay)
    else:
        raise NotImplementedError("optimizer not implemented")
    return optimizer

def get_scheduler(scheduler_name, optimizer, train_dataloader, epochs, batch_size):

    steps_per_epoch = int(len(train_dataloader) / batch_size)  
    scheduler_name = scheduler_name.lower()
    if scheduler_name == "cosine":
        scheduler = CosineWarmupScheduler(
                optimizer,
                warmup_epochs=1,
                max_epochs=epochs * steps_per_epoch,
                warmup_start_lr=1e-1,
                eta_min=1e-5
            )
    else:
        scheduler = None


class CosineWarmupScheduler(_LRScheduler):
    """
    Implements cosine learning rate decay with linear warmup
    
    Args:
        optimizer: wrapped optimizer
        warmup_epochs: number of warmup epochs
        max_epochs: total number of training epochs
        warmup_start_lr: initial learning rate for warmup phase (default: 0)
        eta_min: minimum learning rate (default: 0)
        last_epoch: the index of last epoch (default: -1)
    """
    def __init__(
        self,
        optimizer,
        warmup_epochs,
        max_epochs,
        warmup_start_lr=0.0,
        eta_min=0.0,
        last_epoch=-1,
    ):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.eta_min = eta_min
        
        super(CosineWarmupScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                        "please use `get_last_lr()`.", UserWarning)
            
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup phase
            alpha = self.last_epoch / self.warmup_epochs
            return [self.warmup_start_lr + (base_lr - self.warmup_start_lr) * alpha
                    for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            
            return [self.eta_min + (base_lr - self.eta_min) * cosine_decay
                    for base_lr in self.base_lrs]