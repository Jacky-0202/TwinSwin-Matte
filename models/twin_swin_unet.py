# models/twin_swin_unet.py

import torch
import torch.nn as nn
import timm
from models.blocks import Up, DoubleConv

class TwinSwinUNet(nn.Module):
    def __init__(self, n_classes=1, img_size=384, backbone_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        TwinSwinUNet: A U-Net style architecture using a Swin Transformer Encoder.
        Designed for Locator Matting tasks with Twin-Alignment strategy.
        
        Args:
            n_classes (int): Number of output classes (channels).
            img_size (int): Input image resolution.
            backbone_name (str): Name of the timm Swin Transformer backbone.
            pretrained (bool): Whether to load ImageNet pretrained weights.
        """
        super(TwinSwinUNet, self).__init__()
        
        # --- 1. Encoder (Backbone) ---
        # Initialize Swin Transformer from timm library
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,       # We only need features, not the classifier
            out_indices=(0, 1, 2, 3), # Capture outputs from all 4 stages
            img_size=img_size
        )

        # Store dims as a class attribute (self.dims)
        # This is required so the forward() method can access it for shape correction.
        self.dims = self.backbone.feature_info.channels()
        print(f"[{backbone_name}] Detected feature channels: {self.dims}")

        # Unpack channel dimensions for the Decoder
        c0, c1, c2, c3 = self.dims

        # --- 2. Decoder (Skip Connections) ---
        # U-Net style upsampling and fusion
        
        # Up-Block 1: Fuse Stage 3 (Deepest) with Stage 2
        # Input: c3, Skip: c2 -> Output: c2
        self.up1 = Up(c3, c2, c2)
        
        # Up-Block 2: Fuse Stage 2 with Stage 1
        # Input: c2, Skip: c1 -> Output: c1
        self.up2 = Up(c2, c1, c1)
        
        # Up-Block 3: Fuse Stage 1 with Stage 0
        # Input: c1, Skip: c0 -> Output: c0
        self.up3 = Up(c1, c0, c0)

        # --- 3. Final Expansion (H/4 -> H) ---
        # Swin Stage 0 output is H/4. We need two more upsampling steps to reach full resolution.
        
        # Expand 1: H/4 -> H/2
        self.expand_1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            DoubleConv(c0, c0 // 2)
        )
        
        # Expand 2: H/2 -> H
        self.expand_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(c0 // 2, 24, kernel_size=3, padding=1),
            nn.BatchNorm2d(24),
            nn.ReLU(inplace=True)
        )

        # Head: Final projection to n_classes
        self.head = nn.Conv2d(24, n_classes, kernel_size=1)

    # Added 'expected_c' argument to match the call in forward()
    def _to_nchw(self, x, expected_c):
        """
        Robustly converts (N, H, W, C) to (N, C, H, W).
        
        Args:
            x (Tensor): Input feature map.
            expected_c (int): Expected number of channels for this layer.
        
        Returns:
            Tensor: Feature map in NCHW format.
        """
        # Case 1: Already NCHW (Dimension 1 matches expected channels)
        if x.shape[1] == expected_c:
            return x
        
        # Case 2: NHWC (Last dimension matches expected channels)
        # Swin Transformer usually outputs this format.
        elif x.shape[-1] == expected_c:
            return x.permute(0, 3, 1, 2).contiguous()
        
        # Fallback: If dimensions are ambiguous, force permute assuming NHWC
        else:
            return x.permute(0, 3, 1, 2).contiguous()

    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x (Tensor): Input image (B, 3, H, W)
            
        Returns:
            logits (Tensor): Predicted mask logits.
            features (List): [Optional] Intermediate features for Twin-Alignment training.
        """
        # --- Encoder ---
        features_raw = self.backbone(x)
        
        # Loop through features and fix dimensions
        # using the stored self.dims to verify channel count.
        features = []
        for i, f in enumerate(features_raw):
            features.append(self._to_nchw(f, self.dims[i]))
            
        # Unpack corrected features
        x0, x1, x2, x3 = features

        # --- Decoder ---
        # Perform U-Net skip connections and fusion
        d1 = self.up1(x3, x2) # Fusion of Stage 3 & 2
        d2 = self.up2(d1, x1) # Fusion of Stage 2 & 1
        d3 = self.up3(d2, x0) # Fusion of Stage 1 & 0

        # --- Final Resolution Recovery ---
        d4 = self.expand_1(d3)      # H/4 -> H/2
        d_final = self.expand_2(d4) # H/2 -> H

        # --- Head ---
        logits = self.head(d_final)
        
        # --- Twin-Alignment Strategy ---
        # If in training mode, return intermediate features (x2, x3)
        # to calculate Feature Alignment Loss with the Teacher network.
        if self.training:
            # x2: Stage 2 features (1/16 scale)
            # x3: Stage 3 features (1/32 scale)
            return logits, [x2, x3]
        else:
            return logits