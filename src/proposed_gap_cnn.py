"""PyTorch ProposedGAPCNN from the Group 18 notebook.

Binary convention: logit > 0 predicts hotspot (HS=1); NHS=0.
"""
from torch import nn


class ProposedGAPCNN(nn.Module):
    """Three-convolution CNN with global average pooling for spatial explanations."""
    def __init__(self, input_channels: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 12, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(12, 12, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(12, eps=0.001, momentum=0.01)
        self.elu = nn.ELU()
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(12, 1)

    def forward(self, x):
        x = self.elu(self.conv1(x))
        x = self.elu(self.conv2(x))
        x = self.conv3(x)
        x = self.bn(x)
        x = self.elu(x)
        x = self.gap(x)
        x = x.flatten(1)
        return self.classifier(x)
