# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from config import Config

# ==========================================
# 1. Structure Loss (The Soul of BiRefNet)
#    Focuses on edge-weighted perception.
# ==========================================
class StructureLoss(nn.Module):
    def __init__(self):
        super(StructureLoss, self).__init__()

    def forward(self, pred, mask):
        """
        Args:
            pred: Model Logits (B, 1, H, W) -> Before Sigmoid
            mask: Ground Truth (B, 1, H, W) -> 0~1 Float
        """
        # 1. Generate Edge-Weighted Map (weit)
        wb = 1.0
        target = mask.float()
        
        # Use AvgPool to find edges (High variance areas)
        # padding=15 matches kernel_size=31
        weit = 1 + 5 * torch.abs(F.avg_pool2d(target, kernel_size=31, stride=1, padding=15) - target)
        
        # 2. Weighted BCE Loss
        # Clamp logits for numerical stability to prevent NaN
        pred = torch.clamp(pred, min=-10, max=10) 
        wbce = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

        # 3. Weighted IoU Loss
        pred_prob = torch.sigmoid(pred)
        inter = ((pred_prob * target) * weit).sum(dim=(2, 3))
        union = ((pred_prob + target) * weit).sum(dim=(2, 3))
        wiou = 1 - (inter + 1) / (union - inter + 1)

        # Return average of BCE + IoU
        return (wbce + wiou).mean()

# ==========================================
# 2. Gradient Loss (Sharpness Enforcer)
# ==========================================
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()
        # Sobel Kernels
        kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
        kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
        self.kernel_x = kernel_x.view(1, 1, 3, 3)
        self.kernel_y = kernel_y.view(1, 1, 3, 3)

    def forward(self, pred, gt):
        # Move kernels to same device as input on the fly
        if pred.device != self.kernel_x.device:
            self.kernel_x = self.kernel_x.to(pred.device)
            self.kernel_y = self.kernel_y.to(pred.device)

        # Compute gradients
        pred_grad_x = F.conv2d(pred, self.kernel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.kernel_y, padding=1)
        gt_grad_x = F.conv2d(gt, self.kernel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.kernel_y, padding=1)

        # L1 distance between gradients
        return (torch.abs(pred_grad_x - gt_grad_x) + torch.abs(pred_grad_y - gt_grad_y)).mean()

# ==========================================
# 3. SSIM Loss (Structural Consistency)
# ==========================================
class SSIMLoss(nn.Module):
    def __init__(self, window_size=11, size_average=True):
        super(SSIMLoss, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = self.create_window(window_size, self.channel)

    def gaussian(self, window_size, sigma):
        gauss = torch.Tensor([math.exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
        return gauss/gauss.sum()

    def create_window(self, window_size, channel):
        _1D_window = self.gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
        return window

    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)

        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2

        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2

        C1 = 0.01**2
        C2 = 0.03**2

        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()
        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = self.create_window(self.window_size, channel)
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            self.window = window
            self.channel = channel
        return 1.0 - self._ssim(img1, img2, window, self.window_size, channel, self.size_average)

# ==========================================
# 4. Total Loss Factory (The Adapter)
#    Connects Config -> Model Output -> Loss Components
# ==========================================
class TotalLoss(nn.Module):
    def __init__(self):
        super(TotalLoss, self).__init__()
        
        # Load weights from Config (The Control Center)
        self.weights = Config.LOSS_WEIGHTS
        
        # Initialize components
        self.structure_loss = StructureLoss()
        self.l1 = nn.L1Loss()
        self.grad = GradientLoss()
        self.ssim = SSIMLoss()
        self.mse = nn.MSELoss() # For Feature Alignment

    def forward(self, outputs, target, teacher_feats=None):
        """
        Unified Forward Pass.
        
        Args:
            outputs (dict): {'fine': Logits, 'feats': [Student Features]}
            target (Tensor): GT Mask (B, 1, H, W)
            teacher_feats (list): [Teacher Features] (Optional, passed from train.py)
        
        Returns:
            total_loss (Tensor): Scalar for backward
            loss_dict (dict): Breakdown for logging
        """
        pred_logits = outputs['fine']
        stu_feats = outputs.get('feats', None)
        gt_f32 = target.float()
        
        total_loss = 0.0
        loss_dict = {}

        # --- A. Structure Loss (Uses Logits) ---
        # "weight_struct" corresponds to 'bce' + 'dice' conceptual role in BiRefNet context
        # But let's look for a specific key if defined, or map standard Matting keys.
        # Since Config defines 'l1', 'grad', 'ssim', 'feat', let's stick to those.
        # However, StructureLoss combines BCE and IoU. Let's map it to 'struct' if exists,
        # or treat it as a base component if MODE is 'LOCATOR' or 'MATTING'.
        
        # Let's map 'bce' weight to StructureLoss for simplicity, 
        # or use a new key 'struct' in Config if you prefer.
        # For now, let's assume if we are in MATTING mode, we rely on L1/Grad/SSIM.
        # If we want BiRefNet power, we should probably add 'struct' to Config.
        
        # [AUTO-ADAPTATION]
        # If Config has 'bce' > 0 (Locator), we use StructureLoss as it's better than plain BCE.
        w_struct = self.weights.get('bce', 0.0) + self.weights.get('dice', 0.0) # Combine weights
        if w_struct > 0:
            l_struct = self.structure_loss(pred_logits, gt_f32)
            total_loss += w_struct * l_struct
            loss_dict['struct'] = l_struct.item()

        # --- B. Probability-based Losses (Sigmoid) ---
        pred_prob = torch.sigmoid(pred_logits)

        # 1. L1 Loss
        w_l1 = self.weights.get('l1', 0.0)
        if w_l1 > 0:
            l_l1 = self.l1(pred_prob, gt_f32)
            total_loss += w_l1 * l_l1
            loss_dict['l1'] = l_l1.item()

        # 2. Gradient Loss
        w_grad = self.weights.get('grad', 0.0)
        if w_grad > 0:
            l_grad = self.grad(pred_prob, gt_f32)
            total_loss += w_grad * l_grad
            loss_dict['grad'] = l_grad.item()

        # 3. SSIM Loss
        w_ssim = self.weights.get('ssim', 0.0)
        if w_ssim > 0:
            l_ssim = self.ssim(pred_prob, gt_f32)
            total_loss += w_ssim * l_ssim
            loss_dict['ssim'] = l_ssim.item()

        # --- C. Feature Alignment (Twin Strategy) ---
        w_feat = self.weights.get('feat', 0.0)
        if w_feat > 0 and stu_feats is not None and teacher_feats is not None:
            l_feat = 0.0
            # Align H/16 (Index 0 in list) and H/32 (Index 1 in list)
            # Make sure lists are same length
            min_len = min(len(stu_feats), len(teacher_feats))
            for i in range(min_len):
                l_feat += self.mse(stu_feats[i].float(), teacher_feats[i].float())
            
            total_loss += w_feat * l_feat
            loss_dict['feat'] = l_feat.item()

        return total_loss, loss_dict