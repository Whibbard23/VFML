# Image reader wrapper that logs missing files
# small metric functions: per-event accuracy, F1, Median frame error

"""
Utility helpers: simple model, metrics, and checkpoint helpers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from pathlib import Path

class SimpleCNN(nn.Module):
    def __init__(self, in_channels=3, num_classes=3):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool = nn.MaxPool2d(2,2)
        self.fc1 = nn.Linear(64 * 32 * 32, 256)  # assumes 128x128 input
        self.fc2 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def accuracy(preds, labels):
    with torch.no_grad():
        valid = labels >= 0
        if valid.sum() == 0:
            return 0.0
        p = preds.argmax(dim=1)
        correct = (p[valid] == labels[valid]).sum().item()
        return correct / valid.sum().item()

def save_checkpoint(state, out_dir, name="checkpoint.pth"):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    torch.save(state, path)
    return path

def load_checkpoint(path, device="cpu"):
    if not Path(path).exists():
        return None
    return torch.load(path, map_location=device)
