import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np

class MixAugmentation:
    def __init__(self, alpha=0.2, p_cut=0.2, p_switch = 0.5):
        self.alpha = alpha
        self.p_cut = p_cut
        self.p_switch = p_switch

    def __call__(self, batch): 
        pass

class Mixup:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        
    def __call__(self, batch): 
        images = torch.stack([item[0] for item in batch])
        labels = torch.tensor([item[1] for item in batch]) 
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

class CutMix:
    def __init__(self, p_cut=0.2):
        self.p_cut = p_cut
        
    def __call__(self, batch): 
        pass