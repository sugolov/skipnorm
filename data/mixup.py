import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        batch_size = len(images)
        
        # Generate mixup weights
        weights = np.random.beta(self.alpha, self.alpha, batch_size)
        weights = torch.from_numpy(weights).float()
        
        # Create shuffled indices
        indices = torch.randperm(batch_size)
        
        # Mix the images
        weights = weights.view(-1, 1, 1, 1)
        mixed_images = weights * images + (1 - weights) * images[indices]
        
        # Mix the labels
        weights = weights.view(-1)
        mixed_labels = weights * labels + (1 - weights) * labels[indices]
        
        return mixed_images, mixed_labels