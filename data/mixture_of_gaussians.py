import torch
from torch.utils.data import Dataset, DataLoader

class LabeledGaussianMixtureDataset(Dataset):
    """Dataset class for Gaussian mixture samples with labels."""
    def __init__(self, samples, labels):
        """
        Args:
            samples: tensor of shape (n_samples, dim)
            labels: tensor of shape (n_samples, 1) containing cluster labels
        """
        self.samples = samples
        self.labels = labels
        
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx], self.labels[idx]

def mixture_of_gaussians(n, dim, proportions, mus, sigmas, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    
    # Convert inputs to tensors and validate
    proportions = torch.tensor(proportions)
    mus = [torch.tensor(mu) if not isinstance(mu, torch.Tensor) else mu for mu in mus]
    sigmas = torch.tensor(sigmas)
    
    assert torch.abs(torch.sum(proportions) - 1.0) < 1e-6, "Proportions must sum to 1"
    n_mixtures = len(proportions)
    
    n_samples = [int(n * p) for p in proportions]
    n_samples[-1] = n - sum(n_samples[:-1])
    
    gaussians = []
    labels = []
    for i, (n_comp, mu, sigma) in enumerate(zip(n_samples, mus, sigmas)):
        samples = mu + sigma * torch.randn((int(n_comp), dim))
        gaussians.append(samples)
        labels.append(i * torch.ones(len(samples)))
    
    samples, labels = torch.cat(gaussians, dim=0), torch.cat(labels)
    return LabeledGaussianMixtureDataset(samples, labels)

def two_class_mog_dataloaders(n_train=5000, n_test=500, D=5, batch_size=128):
    train_data = mixture_of_gaussians(
        n = n_train, 
        dim = D, 
        proportions = (0.2, 0.8), 
        mus = ( 0.5 * torch.ones(D),  -0.5 * torch.ones(D)), 
        sigmas = (0.2, 0.2)
    )

    test_data = mixture_of_gaussians(
        n= n_test, 
        dim = D, 
        proportions = (0.2, 0.8), 
        mus = ( 0.5 * torch.ones(D),  -0.5 * torch.ones(D)),
        sigmas = (0.2, 0.2)
    )

    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True
    )

    test_loader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=True
    )

    return train_loader, test_loader