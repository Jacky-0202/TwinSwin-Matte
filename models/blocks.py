# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Convolution => [BN] => ReLU) * 2
    The standard building block of U-Net.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            # First Conv
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, mid_channels),
            nn.ReLU(inplace=True),
            
            # Second Conv
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """
    Upscaling then double conv.
    Updated to handle asymmetric channel sizes from Swin Transformer.
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # If bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Current feature map from decoder (needs upsampling)
        x2: Skip connection feature map from encoder
        """
        x1 = self.up(x1)
        
        # [Expert Logic] Handling Shape Mismatch
        # Input is CHW.
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis (dim=1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class OutConv(nn.Module):
    """
    Final 1x1 convolution to map features to classes.
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
    
class ASPPConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, dilation):
        modules = [
            nn.Conv2d(in_channels, out_channels, 3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        ]
        super(ASPPConv, self).__init__(*modules)

class ASPPPooling(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(ASPPPooling, self).__init__(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        size = x.shape[-2:]
        x = super(ASPPPooling, self).forward(x)
        return F.interpolate(x, size=size, mode='bilinear', align_corners=False)

class ASPP(nn.Module):
    def __init__(self, in_channels, atrous_rates, out_channels=256):
        super(ASPP, self).__init__()
        modules = []
        # 1x1 Conv
        modules.append(nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True)))

        # Atrous Convs
        for rate in atrous_rates:
            modules.append(ASPPConv(in_channels, out_channels, rate))

        # Global Pooling
        modules.append(ASPPPooling(in_channels, out_channels))

        self.convs = nn.ModuleList(modules)

        # Projection
        self.project = nn.Sequential(
            nn.Conv2d(len(modules) * out_channels, out_channels, 1, bias=False),
            nn.GroupNorm(32, out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5)
        )

    def forward(self, x):
        res = []
        for conv in self.convs:
            res.append(conv(x))
        res = torch.cat(res, dim=1)
        return self.project(res)
    
class BasicRFB(nn.Module):
    """
    Receptive Field Block (RFB) - Optimized for Mobile/Real-time Segmentation.
    Inspired by human visual systems, it enhances feature discrimination and robustness.
    
    Key Features:
    - Lightweight: Reduces intermediate channels (1/8 of input).
    - Multi-scale: Uses dilated convolutions with rates [1, 3, 5].
    - BS=1 Support: Uses GroupNorm instead of BatchNorm.
    """
    def __init__(self, in_channels, out_channels, scale=0.1, visual=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_channels
        
        # Reduce intermediate channels for efficiency (Lightweight design)
        inter_channels = in_channels // 8

        # --- Branch 0: 1x1 Conv ---
        # Captures local pointwise information
        self.branch0 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.GroupNorm(4, inter_channels), # GroupNorm for BS=1 safety
            nn.ReLU(inplace=True)
        )

        # --- Branch 1: 3x3 Conv (Dilation Rate 1) ---
        # Standard receptive field
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, dilation=1, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Branch 2: 3x3 Conv (Dilation Rate 3) ---
        # Medium receptive field to capture context
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=3, dilation=3, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Branch 3: 3x3 Conv (Dilation Rate 5) ---
        # Large receptive field for global context
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, inter_channels, 1, bias=False),
            nn.Conv2d(inter_channels, inter_channels, 3, padding=5, dilation=5, bias=False),
            nn.GroupNorm(4, inter_channels), 
            nn.ReLU(inplace=True)
        )

        # --- Feature Fusion ---
        # Concatenate all branches and mix them
        self.conv_cat = nn.Sequential(
            nn.Conv2d(inter_channels * 4, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(32, out_channels), 
            nn.ReLU(inplace=True)
        )
        
        # --- Shortcut Connection ---
        # If input and output channels differ, use a 1x1 conv to align them
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, bias=False),
                nn.GroupNorm(32, out_channels)
            )

    def forward(self, x):
        # Process all branches
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        
        # Concatenate features along the channel dimension
        x_cat = torch.cat((x0, x1, x2, x3), 1)
        
        # Fuse features
        x_cat = self.conv_cat(x_cat)
        
        # Weighted Residual Connection (Scale helps stability)
        x_out = self.shortcut(x) + x_cat * self.scale
        
        return F.relu(x_out, inplace=True)