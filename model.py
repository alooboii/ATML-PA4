"""
CNN Model Architecture
AGREED UPON BY ALL TEAM MEMBERS - DO NOT MODIFY
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CNN(nn.Module):
    """
    CNN Architecture for CIFAR-10
    3 Convolutional Layers + 2 Fully Connected Layers
    Total parameters: ~670K
    """
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 4 * 4, 256)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        # Input: 3x32x32
        x = self.pool(F.relu(self.conv1(x)))  # 32x16x16
        x = self.pool(F.relu(self.conv2(x)))  # 64x8x8
        x = self.pool(F.relu(self.conv3(x)))  # 128x4x4
        x = x.view(-1, 128 * 4 * 4)           # Flatten to 2048
        x = F.relu(self.fc1(x))               # 256
        x = self.fc2(x)                       # 10
        return x


def get_model():
    """Factory function to create a new model instance"""
    return CNN()


def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(f"Model: CNN")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(1, 3, 32, 32)
    out = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")