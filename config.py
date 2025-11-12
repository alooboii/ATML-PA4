"""
Shared configuration for all experiments
AGREED UPON BY ALL TEAM MEMBERS - DO NOT MODIFY WITHOUT COORDINATION
"""

import torch

# ============================================================================
# SHARED CONFIGURATION
# ============================================================================

CONFIG = {
    # ========== TRAINING HYPERPARAMETERS ==========
    'optimizer': 'SGD',
    'lr': 0.01,
    'momentum': 0.9,
    'weight_decay': 1e-4,
    'lr_schedule': None,
    'local_batch_size': 64,  # Adjust for P100: 32/64/128
    'test_batch_size': 128,
    'random_seed': 42,
    
    # ========== FEDERATED LEARNING SETUP ==========
    'num_clients': 5,
    'total_rounds': 50,
    'local_epochs': 5,  # Default K
    'client_participation': 1.0,  # Default: all clients
    'aggregation': 'weighted',
    
    # ========== DATA DISTRIBUTION ==========
    'alpha_iid': 100.0,
    'alpha_moderate': 1.0,
    'alpha_high': 0.1,
    'alpha_extreme': 0.05,
    
    # ========== EVALUATION ==========
    'eval_every_round': True,
    'metrics': ['test_accuracy', 'test_loss'],
    
    # ========== DEVICE ==========
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    
    # ========== DATA ==========
    'dataset': 'CIFAR10',
    'num_classes': 10,
    'data_path': './data',
    
    # ========== NORMALIZATION (CIFAR-10 STANDARD) ==========
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2023, 0.1994, 0.2010],
}


def print_config():
    """Print configuration summary"""
    print("=" * 70)
    print("FEDERATED LEARNING CONFIGURATION")
    print("=" * 70)
    print(f"Device: {CONFIG['device']}")
    print(f"Dataset: {CONFIG['dataset']}")
    print(f"Clients: {CONFIG['num_clients']}")
    print(f"Rounds: {CONFIG['total_rounds']}")
    print(f"Local epochs (K): {CONFIG['local_epochs']}")
    print(f"Batch size: {CONFIG['local_batch_size']}")
    print(f"Learning rate: {CONFIG['lr']}")
    print(f"Random seed: {CONFIG['random_seed']}")
    print("=" * 70)


if __name__ == "__main__":
    print_config()