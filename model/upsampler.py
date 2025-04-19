import torch
import torch.nn as nn
import numpy as np
import torchvision.transforms as T
import torch.nn.functional as F
from typing import Optional
from torchvision.ops import DeformConv2d
from utils.registry import Registry

UPSAMPLER = Registry(name='upsampler', module_key='model.upsampler')

__all__ = ['UPSAMPLER', *UPSAMPLER.registry_names]


class DeformableKernelPredictor(nn.Module):
    def __init__(self, in_channels: int, kernel_size: int = 3):
        """
        Predicts deformable convolution parameters and dynamic kernels

        Args:
            in_channels (int): Number of input feature channels
            kernel_size (int): Spatial size of the predicted convolution kernel
        """
        super().__init__()
        self.coord_att = nn.Sequential(
            nn.Conv2d(in_channels, 64, 1),
            nn.ReLU(),
            nn.Conv2d(64, kernel_size ** 2 * 2, 1)  # Generate (x,y) offset coordinates for each kernel position
        )
        self.kernel_gen = nn.Sequential(
            nn.Conv2d(in_channels, 128, 1),
            nn.ReLU(),
            nn.Conv2d(128, kernel_size ** 2, 1)  # Generate kernel weights for each position
        )
        self.kernel_size = kernel_size

    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass for deformable kernel prediction

        Args:
            x (torch.Tensor): Input feature map with shape (B, C, H, W)

        Returns:
            tuple: (offset, kernel) where
                - offset: Deformation offsets with shape (B, 2, K^2, H, W)
                - kernel: Dynamic convolution weights with shape (B, K^2, H, W)
                (K = kernel_size)
        """
        # Generate spatial offsets for deformable convolution
        offset = self.coord_att(x).reshape(-1, 2, self.kernel_size ** 2, *x.shape[2:])

        # Generate position-dependent convolution kernels
        kernel = self.kernel_gen(x).reshape(-1, self.kernel_size ** 2, *x.shape[2:])
        return offset, kernel


class HighFrequencyExtractor(nn.Module):
    def __init__(self):
        """Laplacian-based high frequency component extractor"""
        super().__init__()

        self.laplacian = nn.Conv2d(3, 3, kernel_size=3, padding=1, bias=False)
        weight = torch.tensor([[0, 1, 0],
                               [1, -4, 1],
                               [0, 1, 0]], dtype=torch.float32)
        # Initialize fixed laplacian kernel (edge detection filter)
        self.laplacian.weight.data = weight.view(1, 1, 3, 3).repeat(3, 3, 1, 1)

    def forward(self, hr_feature: torch.Tensor) -> torch.Tensor:
        """
        Extract high-frequency components using Laplacian operator

        Args:
            hr_feature (torch.Tensor): High-resolution input with shape (B, C, H, W)

        Returns:
            torch.Tensor: High-frequency features with shape (B, C, H, W)
        """
        return self.laplacian(hr_feature)


class HighFrequencyCompensation(nn.Module):
    def __init__(self, in_channels: int):
        """
        High-frequency feature compensation module

        Args:
            in_channels (int): Number of input feature channels
        """
        super().__init__()
        # High-frequency feature extraction
        self.hf_extract = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // 4, in_channels // 4, 3, padding=1),
            nn.ReLU()
        )
        # Residual compensation generator
        self.res_generator = nn.Sequential(
            nn.Conv2d(in_channels // 4, in_channels, 3, padding=1),
            nn.Sigmoid()
        )

        # Learnable high-pass filter initialization
        self.hpf_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, bias=False)
        # Initialize as standard high-pass filter:
        nn.init.constant_(self.hpf_conv.weight, -1 / 8)
        self.hpf_conv.weight.data[:, :, 1, 1] = 1.0  # Center weight=1, surrounding weights=-1/8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Enhance high-frequency components through residual compensation

        Args:
            x (torch.Tensor): Input features with shape (B, C, H, W)

        Returns:
            torch.Tensor: Enhanced features with shape (B, C, H, W)
        """
        # Extract high-frequency components
        hf = self.hpf_conv(x)
        # Generate adaptive compensation weights
        comp = self.hf_extract(hf)
        comp = self.res_generator(comp)

        return x * (1 + comp)


class DeformableAlignmentFusion(nn.Module):
    def __init__(self, in_channels: int):
        """
        Feature alignment module using deformable convolution

        Args:
            in_channels (int): Number of input feature channels
        """
        super().__init__()
        # Offset prediction
        self.offset_conv = nn.Conv2d(in_channels * 2, 2 * 3 * 3, 3, padding=1)
        self.deform_conv = DeformConv2d(in_channels, in_channels, 3, padding=1)

        # Feature modulation gate
        self.modulation = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, 1),
            nn.Sigmoid()
        )

    def forward(self, low_res: torch.Tensor, high_res: torch.Tensor) -> torch.Tensor:
        """
        Align and fuse features using deformable convolution

        Args:
            low_res (torch.Tensor): Low-resolution upsampled features (B, C, H, W)
            high_res (torch.Tensor): High-resolution features (B, C, H, W)

        Returns:
            torch.Tensor: Fused features with shape (B, C, H, W)
        """
        # Concatenate features for offset prediction
        concat_feat = torch.cat([low_res, high_res], dim=1)
        offset = self.offset_conv(concat_feat)

        # Perform deformable alignment
        aligned_feat = self.deform_conv(low_res, offset)

        # Generate fusion gate
        gate = self.modulation(concat_feat)

        return aligned_feat * gate + high_res


@UPSAMPLER.register()
class HFGDU(nn.Module):
    def __init__(self, in_channels: int, scale_factor: int = 2):
        """
        High-Frequency Guided Deformable Upsampler

        Args:
            in_channels (int): Number of input feature channels
            scale_factor (int): Spatial upsampling ratio
        """
        super().__init__()
        self.scale = scale_factor

        self.initial_upsample = nn.Upsample(
            scale_factor=scale_factor,
            mode='bilinear',
            align_corners=True
        )
        self.hf_comp = HighFrequencyCompensation(in_channels)
        self.align_fusion = DeformableAlignmentFusion(in_channels)

    def forward(self, x: torch.Tensor, high_res_feat: torch.Tensor) -> torch.Tensor:
        """
        Upsample with high-frequency guidance

        Args:
            x (torch.Tensor): Low-resolution input features (B, C, H, W)
            high_res_feat (torch.Tensor): High-resolution guidance features (B, C, H*scale, W*scale)

        Returns:
            torch.Tensor: Upsampled features with shape (B, C, H*scale, W*scale)
        """
        # Initial interpolation-based upsampling
        up_feat = self.initial_upsample(x)

        # High-frequency detail compensation
        comp_feat = self.hf_comp(up_feat)

        # Deformable alignment with high-res features
        fused_feat = self.align_fusion(comp_feat, high_res_feat)

        return fused_feat

# CARAFE
@UPSAMPLER.register()
class CARAFE(nn.Module):
    def __init__(self, c, c_mid=64, scale=2, k_up=5, k_enc=3):
        """ The implementation of the CARAFE module.

        The details are in "https://arxiv.org/abs/1905.02188".

        Args:
            c: The channel number of the input and the output.
            c_mid: The channel number after compression.
            scale: The expected upsample scale.
            k_up: The size of the reassembly kernel.
            k_enc: The kernel size of the encoder.

        Returns:
            X: The upsampled feature map.
        """
        super(CARAFE, self).__init__()
        self.scale = scale

        self.comp = ConvBNReLU(c, c_mid, kernel_size=1, stride=1,
                               padding=0, dilation=1)
        self.enc = ConvBNReLU(c_mid, (scale * k_up) ** 2, kernel_size=k_enc,
                              stride=1, padding=k_enc // 2, dilation=1,
                              use_relu=False)
        self.pix_shf = nn.PixelShuffle(scale)

        self.upsmp = nn.Upsample(scale_factor=scale, mode='nearest')
        self.unfold = nn.Unfold(kernel_size=k_up, dilation=scale,
                                padding=k_up // 2 * scale)

    def forward(self, X):
        b, c, h, w = X.size()
        h_, w_ = h * self.scale, w * self.scale

        W = self.comp(X)  # b * m * h * w
        W = self.enc(W)  # b * 100 * h * w
        W = self.pix_shf(W)  # b * 25 * h_ * w_
        W = F.softmax(W, dim=1)  # b * 25 * h_ * w_

        X = self.upsmp(X)  # b * c * h_ * w_
        X = self.unfold(X)  # b * 25c * h_ * w_
        X = X.view(b, c, -1, h_, w_)  # b * 25 * c * h_ * w_

        X = torch.einsum('bkhw,bckhw->bchw', [W, X])  # b * c * h_ * w_
        return X

class ConvBNReLU(nn.Module):
    '''Module for the Conv-BN-ReLU tuple.'''

    def __init__(self, c_in, c_out, kernel_size, stride, padding, dilation,
                 use_relu=True):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(
            c_in, c_out, kernel_size=kernel_size, stride=stride,
            padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(c_out)
        if use_relu:
            self.relu = nn.ReLU(inplace=True)
        else:
            self.relu = None

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

#TransposeConvUpsample
@UPSAMPLER.register()
class TransposeConvUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1):
        super(TransposeConvUpsample, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size,
                                             stride=stride, padding=padding, output_padding=output_padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.trans_conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


def normal_init(module, mean=0, std=1, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.normal_(module.weight, mean, std)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def constant_init(module, val, bias=0):
    if hasattr(module, 'weight') and module.weight is not None:
        nn.init.constant_(module.weight, val)
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.constant_(module.bias, bias)

@UPSAMPLER.register()
class DySample(nn.Module):
    """
    This code is from https://github.com/tiny-smart/dysample/blob/main/dysample.py
    """
    def __init__(self, in_channels, scale=2, style='lp', groups=4, dyscope=False):
        super().__init__()
        self.scale = scale
        self.style = style
        self.groups = groups
        assert style in ['lp', 'pl']
        if style == 'pl':
            assert in_channels >= scale ** 2 and in_channels % scale ** 2 == 0
        assert in_channels >= groups and in_channels % groups == 0

        if style == 'pl':
            in_channels = in_channels // scale ** 2
            out_channels = 2 * groups
        else:
            out_channels = 2 * groups * scale ** 2

        self.offset = nn.Conv2d(in_channels, out_channels, 1)
        normal_init(self.offset, std=0.001)
        if dyscope:
            self.scope = nn.Conv2d(in_channels, out_channels, 1, bias=False)
            constant_init(self.scope, val=0.)

        self.register_buffer('init_pos', self._init_pos())

    def _init_pos(self):
        h = torch.arange((-self.scale + 1) / 2, (self.scale - 1) / 2 + 1) / self.scale
        return torch.stack(torch.meshgrid([h, h])).transpose(1, 2).repeat(1, self.groups, 1).reshape(1, -1, 1, 1)

    def sample(self, x, offset):
        B, _, H, W = offset.shape
        offset = offset.view(B, 2, -1, H, W)
        coords_h = torch.arange(H) + 0.5
        coords_w = torch.arange(W) + 0.5
        coords = torch.stack(torch.meshgrid([coords_w, coords_h])
                             ).transpose(1, 2).unsqueeze(1).unsqueeze(0).type(x.dtype).to(x.device)
        normalizer = torch.tensor([W, H], dtype=x.dtype, device=x.device).view(1, 2, 1, 1, 1)
        coords = 2 * (coords + offset) / normalizer - 1
        coords = F.pixel_shuffle(coords.view(B, -1, H, W), self.scale).view(
            B, 2, -1, self.scale * H, self.scale * W).permute(0, 2, 3, 4, 1).contiguous().flatten(0, 1)
        return F.grid_sample(x.reshape(B * self.groups, -1, H, W), coords, mode='bilinear',
                             align_corners=False, padding_mode="border").view(B, -1, self.scale * H, self.scale * W)

    def forward_lp(self, x):
        if hasattr(self, 'scope'):
            offset = self.offset(x) * self.scope(x).sigmoid() * 0.5 + self.init_pos
        else:
            offset = self.offset(x) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward_pl(self, x):
        x_ = F.pixel_shuffle(x, self.scale)
        if hasattr(self, 'scope'):
            offset = F.pixel_unshuffle(self.offset(x_) * self.scope(x_).sigmoid(), self.scale) * 0.5 + self.init_pos
        else:
            offset = F.pixel_unshuffle(self.offset(x_), self.scale) * 0.25 + self.init_pos
        return self.sample(x, offset)

    def forward(self, x):
        if self.style == 'pl':
            return self.forward_pl(x)
        return self.forward_lp(x)



if __name__ == '__main__':
    pass

    """
        An example for using the HFGDU module
    """

    # B = 4
    # C = 64
    # H, W = 32, 32
    # scale_factor = 2
    #
    # upsampler = HFGDU(in_channels=C, scale_factor=scale_factor)
    #
    # low_res_feat = torch.randn(B, C, H, W)
    # high_res_feat = torch.randn(B, C, H * scale_factor, W * scale_factor)
    #
    # output = upsampler(low_res_feat, high_res_feat)
    # print(output)