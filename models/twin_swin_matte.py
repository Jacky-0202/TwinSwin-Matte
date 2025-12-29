# models/twin_swin_matte.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from models.blocks import DoubleConv, OutConv, BasicRFB

def load_pretrained_1ch(model, backbone_name):
    """
    Helper function to load ImageNet pretrained weights for a 1-channel backbone.
    """
    try:
        temp_model = timm.create_model(backbone_name, pretrained=True, in_chans=3)
        state_dict = temp_model.state_dict()
        
        if 'patch_embed.proj.weight' in state_dict:
            weight_3ch = state_dict['patch_embed.proj.weight']
            weight_1ch = weight_3ch.mean(dim=1, keepdim=True) 
            state_dict['patch_embed.proj.weight'] = weight_1ch
            
        model.load_state_dict(state_dict, strict=False)
        print(f"✅ [gt_encoder] Initialized 1-channel backbone ({backbone_name}) with averaged ImageNet weights.")
    except Exception as e:
        print(f"⚠️ Warning: Failed to load pretrained weights for gt_encoder: {e}")

class TwinSwinMatteNet(nn.Module):
    def __init__(self, n_classes=1, img_size=224, backbone_name='swin_tiny_patch4_window7_224', pretrained=True):
        """
        TwinSwin-Matte (Lightweight Decoder - 64 Channels).
        
        Architecture Flow:
        1. Adapters:
           - img_encoder features (c1, c3) are passed through 1x1 Conv Adapters before Loss calculation.
           
        2. Decoder (Unified 64-channel width):
           - Path A: c4 -> 1x1 Conv (64ch) -> Upsample
           - Path B: c3 -> RFB (64ch)
           - Deep Fusion: Concat(Path A, Path B) -> DoubleConv (64ch)
           - Path C: c1 -> 1x1 Conv (64ch)
           - Final Fusion: Concat(Deep_Up, Path C) -> DoubleConv (64ch) -> Output (1ch)
        """
        super(TwinSwinMatteNet, self).__init__()
        
        # ==========================================
        # 1. img_encoder Backbone
        # ==========================================
        print(f"🏗️ [img_encoder] Building Backbone: {backbone_name}")
        self.img_encoder = timm.create_model(
            backbone_name, 
            pretrained=pretrained, 
            features_only=True,
            out_indices=(0, 1, 2, 3), 
            img_size=img_size,
            strict_img_size=False
        )
        
        # ==========================================
        # 2. gt_encoder Backbone
        # ==========================================
        print(f"🏗️ [gt_encoder] Building Backbone: {backbone_name} (1-channel)")
        self.gt_encoder = timm.create_model(
            backbone_name, 
            pretrained=False, 
            in_chans=1,
            features_only=True,
            out_indices=(0, 1, 2), 
            img_size=img_size,
            strict_img_size=False
        )
        if pretrained:
            load_pretrained_1ch(self.gt_encoder, backbone_name)
        
        for param in self.gt_encoder.parameters():
            param.requires_grad = False

        # ==========================================
        # 3. Setup Channels
        # ==========================================
        if 'tiny' in backbone_name or 'small' in backbone_name:
            embed_dim = 96
        elif 'base' in backbone_name:
            embed_dim = 128
        elif 'large' in backbone_name:
            embed_dim = 192
        else:
            embed_dim = 96

        self.dims = [embed_dim * (2 ** i) for i in range(4)]
        c1_in = self.dims[0]  # Stride 4
        c3_in = self.dims[2]  # Stride 16
        c4_in = self.dims[3]  # Stride 32
        
        # Define Decoder Width (Lightweight)
        DECODER_DIM = 64

        # ==========================================
        # 4. Feature Adapters (For Loss Only)
        # ==========================================
        # These 1x1 convs adapt img_encoder features to gt_encoder domain for loss calculation.
        # We keep them separate from the decoder path to allow flexible learning.
        self.adapt_c1 = nn.Sequential(
            nn.Conv2d(c1_in, c1_in, 1, bias=False), 
            nn.GroupNorm(16, c1_in),
            nn.ReLU(inplace=True)
        )
        self.adapt_c3 = nn.Sequential(
            nn.Conv2d(c3_in, c3_in, 1, bias=False),
            nn.GroupNorm(32, c3_in),
            nn.ReLU(inplace=True)
        )

        # ==========================================
        # 5. Lightweight Decoder (64 Channels)
        # ==========================================
        
        # --- Deep Branch (c3 + c4) ---
        
        # RFB: c3 (Stride 16) -> 64 channels
        print(f"✨ Using RFB with output: {DECODER_DIM} channels")
        self.neck = BasicRFB(in_channels=c3_in, out_channels=DECODER_DIM)
        
        # Project c4 (Stride 32) -> 64 channels
        self.c4_project = nn.Sequential(
            nn.Conv2d(c4_in, DECODER_DIM, 1, bias=False),
            nn.GroupNorm(16, DECODER_DIM), # 64 channels / 16 groups = 4 ch/group
            nn.ReLU(inplace=True)
        )
        
        # Deep Fusion Block
        # Input: 64 (RFB) + 64 (c4_project) = 128
        # Output: 64
        self.fusion_deep = DoubleConv(in_channels=DECODER_DIM*2, out_channels=DECODER_DIM, mid_channels=DECODER_DIM)

        # --- Shallow Branch (Deep + c1) ---
        
        # Project c1 (Stride 4) -> 64 channels
        self.low_level_project = nn.Sequential(
            nn.Conv2d(c1_in, DECODER_DIM, 1, bias=False),
            nn.GroupNorm(16, DECODER_DIM), 
            nn.ReLU(inplace=True)
        )
        
        # Final Decoder Head
        # Input: 64 (Deep_Up) + 64 (c1_project) = 128
        # Output: 64
        self.decoder_head = DoubleConv(in_channels=DECODER_DIM*2, out_channels=DECODER_DIM, mid_channels=DECODER_DIM)
        
        # Output Head: 64 -> 1
        self.outc = OutConv(DECODER_DIM, n_classes)

    def forward(self, x, gt_mask=None):
        input_shape = x.shape[-2:] # (H, W)
        
        # --- A. img_encoder Stream ---
        stu_raw = self.img_encoder(x)
        stu_feats = []
        for f in stu_raw:
            if len(f.shape) == 4 and f.shape[1] != f.shape[-1]: 
               f = f.permute(0, 3, 1, 2).contiguous()
            stu_feats.append(f)
            
        c1 = stu_feats[0] # Stride 4
        c3 = stu_feats[2] # Stride 16
        c4 = stu_feats[3] # Stride 32
        
        # Apply Adapters (For Loss Calculation)
        c1_adapt = self.adapt_c1(c1)
        c3_adapt = self.adapt_c3(c3)

        # --- B. gt_encoder Stream (Targets) ---
        gt_encoder_c1 = None 
        gt_encoder_c3 = None

        if self.training and gt_mask is not None:
            if gt_mask.dim() == 3: gt_input = gt_mask.unsqueeze(1)
            else: gt_input = gt_mask
            
            if gt_input.max() > 1.0: gt_input = gt_input.float() / 255.0
            else: gt_input = gt_input.float()

            with torch.no_grad():
                tea_raw = self.gt_encoder(gt_input)
                tea_feats = []
                for f in tea_raw:
                    if len(f.shape) == 4 and f.shape[1] != f.shape[-1]:
                        f = f.permute(0, 3, 1, 2).contiguous()
                    tea_feats.append(f)
                
                # Raw gt_encoder Features for Supervision
                gt_encoder_c1 = tea_feats[0]
                gt_encoder_c3 = tea_feats[2] 

        # --- C. Lightweight Decoder ---
        
        # 1. Deep Fusion (c3 & c4)
        x_rfb = self.neck(c3) # (N, 64, H/16, W/16)
        
        x_c4 = self.c4_project(c4) # (N, 64, H/32, W/32)
        x_c4_up = F.interpolate(x_c4, size=x_rfb.shape[2:], mode='bilinear', align_corners=True)
        
        x_deep_cat = torch.cat([x_rfb, x_c4_up], dim=1) # 64 + 64 = 128
        x_deep_fused = self.fusion_deep(x_deep_cat)     # -> 64
        
        # 2. Shallow Fusion (Deep & c1)
        x_up_mid = F.interpolate(x_deep_fused, size=c1.shape[2:], mode='bilinear', align_corners=True)
        
        c1_low = self.low_level_project(c1) # (N, 64, H/4, W/4)
        
        x_cat = torch.cat([x_up_mid, c1_low], dim=1) # 64 + 64 = 128
        
        x_decode = self.decoder_head(x_cat) # -> 64
        
        # 3. Output
        logits = self.outc(x_decode) # 64 -> 1
        
        pred_alpha = F.interpolate(logits, size=input_shape, mode='bilinear', align_corners=True)
        pred_alpha = torch.sigmoid(pred_alpha)

        return pred_alpha, (c3_adapt, c1_adapt), (gt_encoder_c3, gt_encoder_c1)