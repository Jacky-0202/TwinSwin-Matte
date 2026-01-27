# models/blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv2d => BN => ReLU) * 2
    Standard U-Net building block.
    """
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
            
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class Up(nn.Module):
    """
    Upscaling block that handles skip connections.
    Includes padding logic to handle odd-sized feature maps (asymmetric shapes).
    """
    def __init__(self, in_channels, skip_channels, out_channels, bilinear=True):
        super().__init__()

        # Use bilinear upsampling to reduce channel count before concatenation
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            # The input to DoubleConv will be (in_channels//2 + skip_channels)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x1, x2):
        """
        x1: Current feature map (needs upsampling)
        x2: Skip connection from encoder
        """
        x1 = self.up(x1)
        
        # [Padding Logic]
        # Calculate difference in shape if x1 and x2 dimensions don't match exactly
        # (Common with CNN/Transformer pooling operations)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate along channel axis
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)