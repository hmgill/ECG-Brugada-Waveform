# agent/models/resnet1d.py
"""ResNet-1D architecture for ECG classification."""

import torch
import torch.nn as nn


class ResidualBlock1D(nn.Module):
    """Basic residual block for 1D signals."""
    
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super().__init__()
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=7, 
                               stride=stride, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=7,
                               stride=1, padding=3, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """
    ResNet-1D for ECG classification.
    
    Simpler and more stable than Inception for small medical datasets.
    """
    
    def __init__(
        self, 
        in_channels=12, 
        num_classes=1,
        base_filters=64,
        num_blocks=[2, 2, 2, 2],  # 4 stages with 2 blocks each
        dropout=0.3
    ):
        super().__init__()
        
        self.in_channels = base_filters
        
        # Initial convolution
        self.conv1 = nn.Conv1d(in_channels, base_filters, kernel_size=15, 
                               stride=2, padding=7, bias=False)
        self.bn1 = nn.BatchNorm1d(base_filters)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual stages
        self.layer1 = self._make_layer(base_filters, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(base_filters * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(base_filters * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(base_filters * 8, num_blocks[3], stride=2)
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        
        # Classification head
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(base_filters * 8, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channels, out_channels, kernel_size=1, 
                         stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock1D(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock1D(out_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        
        return x
    
    def get_num_params(self):
        return sum(p.numel() for p in self.parameters())


def create_resnet1d(model_size='small', in_channels=12, num_classes=1, dropout=0.3):
    """
    Factory function for ResNet-1D models.
    
    Args:
        model_size: 'small', 'medium', or 'large'
        in_channels: Number of input channels (12 for ECG)
        num_classes: Number of output classes
        dropout: Dropout rate
    """
    configs = {
        'small': {
            'base_filters': 32,
            'num_blocks': [2, 2, 2, 2]  # ~0.5M params
        },
        'medium': {
            'base_filters': 64,
            'num_blocks': [2, 2, 2, 2]  # ~2M params
        },
        'large': {
            'base_filters': 64,
            'num_blocks': [3, 4, 6, 3]  # ~5M params, like ResNet-34
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"model_size must be one of {list(configs.keys())}")
    
    config = configs[model_size]
    
    return ResNet1D(
        in_channels=in_channels,
        num_classes=num_classes,
        base_filters=config['base_filters'],
        num_blocks=config['num_blocks'],
        dropout=dropout
    )
