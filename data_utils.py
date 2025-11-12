"""
Data loading and partitioning utilities for Federated Learning
"""

import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
from collections import defaultdict
from config import CONFIG


def set_seed(seed=42):
    """Set random seeds for reproducibility"""
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_transforms():
    """Get CIFAR-10 train and test transforms"""
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=CONFIG["mean"], std=CONFIG["std"]),
        ]
    )

    return train_transform, test_transform


def load_cifar10():
    """Load CIFAR-10 dataset"""
    train_transform, test_transform = get_transforms()

    trainset = torchvision.datasets.CIFAR10(
        root=CONFIG["data_path"], train=True, download=True, transform=train_transform
    )

    testset = torchvision.datasets.CIFAR10(
        root=CONFIG["data_path"], train=False, download=True, transform=test_transform
    )

    return trainset, testset


def dirichlet_partition(dataset, num_clients, alpha, seed=42):
    """
    Partition dataset using Dirichlet distribution

    Args:
        dataset: PyTorch dataset
        num_clients: Number of clients
        alpha: Dirichlet concentration parameter
               - Large alpha (e.g., 100): IID
               - Small alpha (e.g., 0.1): Highly non-IID
        seed: Random seed

    Returns:
        client_indices: List of lists, each containing indices for one client
    """
    set_seed(seed)

    # Get labels
    labels = np.array([dataset[i][1] for i in range(len(dataset))])
    num_classes = len(np.unique(labels))

    # Group indices by class
    class_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        class_indices[label].append(idx)

    # Convert to numpy arrays
    for label in class_indices:
        class_indices[label] = np.array(class_indices[label])

    # Initialize client indices
    client_indices = [[] for _ in range(num_clients)]

    # For each class, distribute samples to clients using Dirichlet
    for label in range(num_classes):
        indices = class_indices[label]
        np.random.shuffle(indices)

        # Sample from Dirichlet distribution
        proportions = np.random.dirichlet(alpha=[alpha] * num_clients)

        # Ensure proportions sum to 1
        proportions = proportions / proportions.sum()

        # Split indices according to proportions
        proportions = (np.cumsum(proportions) * len(indices)).astype(int)[:-1]
        split_indices = np.split(indices, proportions)

        # Assign to clients
        for client_id, client_split in enumerate(split_indices):
            client_indices[client_id].extend(client_split.tolist())

    # Shuffle each client's indices
    for client_id in range(num_clients):
        np.random.shuffle(client_indices[client_id])

    # Print distribution statistics
    print(f"\nData partitioning (α={alpha}):")
    print(f"  Total samples: {len(dataset)}")
    for client_id in range(num_clients):
        print(f"  Client {client_id}: {len(client_indices[client_id])} samples")

    return client_indices


def get_client_loaders(trainset, client_indices, batch_size):
    """
    Create data loaders for each client

    Args:
        trainset: Training dataset
        client_indices: List of indices for each client
        batch_size: Batch size for training

    Returns:
        client_loaders: List of DataLoader objects
    """
    client_loaders = []

    for indices in client_indices:
        subset = torch.utils.data.Subset(trainset, indices)
        loader = torch.utils.data.DataLoader(
            subset, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True
        )
        client_loaders.append(loader)

    return client_loaders


def get_test_loader(testset, batch_size=128):
    """Create test data loader"""
    test_loader = torch.utils.data.DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True
    )
    return test_loader


if __name__ == "__main__":
    # Test data loading and partitioning
    set_seed(42)
    trainset, testset = load_cifar10()

    print(f"Training set size: {len(trainset)}")
    print(f"Test set size: {len(testset)}")

    # Test Dirichlet partitioning
    client_indices = dirichlet_partition(trainset, num_clients=5, alpha=0.1)

    # Test loader creation
    client_loaders = get_client_loaders(trainset, client_indices, batch_size=32)
    print(f"\nCreated {len(client_loaders)} client loaders")
