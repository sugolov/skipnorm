import torch

from mixup import Mixup

def get_dataloaders(data_name, mixup_alpha=0.2, num_workers=0, **kwargs):
    if data_name == "cifar10":
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
        train_dataset = datasets.CIFAR10(root='./data', train=True,
                                       download=True, transform=transform_train)
        test_dataset = datasets.CIFAR10(root='./data', train=False,
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
    
    return train_dataloader, test_dataloader, num_classes, image_size