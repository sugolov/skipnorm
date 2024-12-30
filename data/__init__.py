import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms

from data.augment import Mixup
from data.mixture_of_gaussians import two_class_mog_dataloaders

def get_dataloaders(data_set, batch_size, mixup_alpha, num_workers=0, data_path="~/.data", pin_memory=False, distributed=False):
    data_set = data_set.lower()
    if data_set == "cifar10":
        # normalized by global mean/std
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.GaussianBlur(kernel_size=3),  # Added Gaussian blur
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                                 (0.2023, 0.1994, 0.2010))
        ])
        
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), 
                               (0.2023, 0.1994, 0.2010))
        ])

        if distributed:
            train_sampler = DistributedSampler(
                trainset,
                num_replicas=world_size,
                rank=rank,
                shuffle=True
            )
            test_sampler = DistributedSampler(
                testset,
                num_replicas=world_size,
                rank=rank,
                shuffle=False
            )
        else:
            train_sampler = None
            test_sampler = None

        # Load datasets
        train_dataset = datasets.CIFAR10(root=data_path, train=True, download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_path, train=False, download=True, transform=transform_test)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=(train_sampler is None),  
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True,
            collate_fn=Mixup(alpha=mixup_alpha)
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size,
            shuffle=False,
            sampler=test_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        num_classes = 10
        image_size = 32
    elif data_set == "two_class_mog":
        return two_class_mog_dataloaders()
    else:
        raise(NotImplementedError("Dataloader not implemented yet"))
    
    return train_dataloader, test_dataloader, num_classes, image_size