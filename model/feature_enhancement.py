import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
from utils.registry import Registry

FEATURE_ENHANCEMENT = Registry(name='feature_enhancement', module_key='model.feature_enhancement')

__all__ = ['FEATURE_ENHANCEMENT', *FEATURE_ENHANCEMENT .registry_names]

@FEATURE_ENHANCEMENT.register()
class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        y = torch.cat([max_pool, avg_pool], dim=1)
        y = self.conv(y)
        return x * self.sigmoid(y)

@FEATURE_ENHANCEMENT.register()
class EnhancedSpatialAttention(nn.Module):
    def __init__(self):
        """
        Dual-Context Progressive Attention Network

        Architecture Components:
        - Base spatial attention encoding
        - Multi-scale feature extraction branch
        - Local contrast enhancement branch
        """
        super().__init__()
        # Multi-scale feature extraction with different dilation rates
        self.dilated_conv3 = nn.Conv2d(2, 32, 3, padding=2, dilation=2)
        self.dilated_conv5 = nn.Conv2d(2, 32, 3, padding=4, dilation=4)
        self.conv1x1 = nn.Conv2d(64, 1, 1)

        # Local contrast enhancement branch
        self.contrast_conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),  # Non-linear activation
            nn.Conv2d(16, 1, 3, padding=1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for spatial attention computation

        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W)

        Returns:
            torch.Tensor: Attention-weighted features with shape (B, C, H, W)
        """
        # Base spatial attention encoding
        max_pool = torch.max(x, dim=1, keepdim=True)[0]  # (B, 1, H, W)
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # (B, 1, H, W)
        spatial_feat = torch.cat([max_pool, avg_pool], dim=1)  # (B, 2, H, W)

        # Multi-scale context branch
        d3 = self.dilated_conv3(spatial_feat)  # small-range context (B, 32, H, W)
        d5 = self.dilated_conv5(spatial_feat)  # large-range context (B, 32, H, W)
        multi_scale = torch.cat([d3, d5], dim=1)  # (B, 64, H, W)
        scale_att = self.conv1x1(multi_scale)  # (B, 1, H, W)

        # Local contrast branch
        contrast = max_pool - F.avg_pool2d(max_pool, 3, stride=1, padding=1)  # Local contrast computation
        contrast_att = self.contrast_conv(torch.abs(contrast))  # (B, 1, H, W)

        combined_att = scale_att + contrast_att
        return x * self.sigmoid(combined_att)

@FEATURE_ENHANCEMENT.register()
class ECA(nn.Module):
    "This code is from https://github.com/BangguWu/ECANet/blob/master/models/eca_module.py."
    """Constructs a ECA module.

    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """
    def __init__(self, channel, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)

        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)

        # Multi-scale information fusion
        y = self.sigmoid(y)

        return x * y.expand_as(x)

if __name__ == '__main__':
    pass

    """
        An example for using it.
    """

    # batch_size = 4
    # channels = 256
    # height = 64
    # width = 64
    #
    # input_tensor = torch.randn(batch_size, channels, height, width)
    # attention_module = EnhancedSpatialAttention()
    #
    # output = attention_module(input_tensor)



