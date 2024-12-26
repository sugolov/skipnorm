import torch
from torchvision import datasets, transforms

from data.mixup import Mixup
from data.mixture_of_gaussians import two_class_mog_dataloaders

def get_dataloaders(data_set, mixup_alpha=0.2, num_workers=0, data_path="~/.data"):
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

        # Load datasets
        train_dataset = datasets.CIFAR10(root=data_path, train=True,
                                       download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root=data_path, train=False,
                                      download=True, transform=transform_test)

        # Create dataloaders
        train_dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=num_workers, 
            collate_fn=Mixup(alpha=mixup_alpha)
        )
        
        test_dataloader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=num_workers
        )

        num_classes = 10
        image_size = 32
    elif data_set == "two_class_mog":
        return two_class_mog_dataloaders()
    else:
        raise(NotImplementedError("Dataloader not implemented yet"))
    
    return train_dataloader, test_dataloader, num_classes, image_size