import torch

import torch.nn as nn
import numpy as np
import torchvision.transforms as T

from typing import Optional

from .register import MODELS
from .hourglass import get_large_hourglass_net

@MODELS.register()
class CrowdSatNet(nn.Module):
    def __init__(
            self,
            num_classes: int = 2,
            head_conv: int = 2
    ):
        """
            :param num_classes: number of output classes, background included
            :param head_conv: number of supplementary convolutional layers at the end of decoder. Defaults to 64.
            :return:
        """

        super(CrowdSatNet, self).__init__()

        self.num_classes = num_classes
        self.head_conv = head_conv

        # head
        heads = {'hm': num_classes-1, 'cl':2}

        self.model = get_large_hourglass_net(heads=heads, pretrained=False, input_size=256)

        self.loc_head = nn.Sequential(
            nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=3, stride=1, padding=1,
                      bias=False)

        )

    def forward(self, input: torch.Tensor):

        encode = self.model(input)

        heatmap_logits = [encode[i]['hm'] for i in range(len(encode))]

        heatmaps = [self.loc_head(heatmap_logit) for heatmap_logit in heatmap_logits]
        # clsmap = self.cls_head(clsmap_logits)

        return heatmaps

    def _freeze_layer(self, layer_name: str) -> None:
        for param in getattr(self, layer_name).parameters():
            param.requires_grad = False