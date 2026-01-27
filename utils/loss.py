import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# --- 1. SSIM Loss (Structural Similarity) ---
class SSIMLoss(nn.Module):
    """
    Calculates the Structural Similarity Index (SSIM) Loss.
    """
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

# --- 2. IoU Loss (Intersection over Union) ---
class IoULoss(nn.Module):
    """
    Calculates the Soft IoU Loss using probabilities.
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth

    def forward(self, pred, gt):
        # inputs are already probabilities [0, 1]
        intersection = (pred * gt).sum(dim=(2, 3))
        total = (pred + gt).sum(dim=(2, 3))
        union = total - intersection
        
        iou = (intersection + self.smooth) / (union + self.smooth)
        return 1.0 - iou.mean()

# --- 3. Main Loss Wrapper (Fixed for Logits & AMP) ---
class MattingLoss(nn.Module):
    """
    Composite Loss function for Locator Matting.
    Handles Logits inputs correctly for mixed precision training.
    """
    def __init__(self, 
                 weight_bce=1.0,    
                 weight_iou=1.0,    
                 weight_ssim=0.5,   
                 weight_l1=0.2,     
                 weight_focal=0.0,  
                 weight_grad=0.0,   
                 weight_feat=0.2):  
        super(MattingLoss, self).__init__()
        
        self.weight_bce = weight_bce
        self.weight_iou = weight_iou
        self.weight_ssim = weight_ssim
        self.weight_l1 = weight_l1
        self.weight_focal = weight_focal # (Not used in this fix, kept for compatibility)
        self.weight_grad = weight_grad
        self.weight_feat = weight_feat
        
        # [FIX] Use BCEWithLogitsLoss for numerical stability with AMP
        self.bce_logits = nn.BCEWithLogitsLoss()
        
        self.iou = IoULoss()
        self.ssim = SSIMLoss(window_size=11)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(self, pred_logits, gt_alpha, stu_feats=None, tea_feats=None):
        """
        Args:
            pred_logits (Tensor): Raw model output (Logits), NOT probabilities. (B, 1, H, W)
            gt_alpha (Tensor): Ground Truth mask (0 or 1). (B, 1, H, W)
            stu_feats, tea_feats: Feature maps for alignment.
        """
        
        # 1. Prepare Ground Truth
        gt_f32 = gt_alpha.float()
        
        # 2. Calculate BCE Loss (Uses Logits directly)
        # This is safe for Autocast
        loss_bce = 0.0
        if self.weight_bce > 0:
            loss_bce = self.bce_logits(pred_logits, gt_f32)
        
        # 3. Generate Probabilities for other losses
        # Sigmoid is required for IoU, SSIM, L1
        pred_prob = torch.sigmoid(pred_logits)
        
        # Clamp for numerical safety in other losses (optional but good practice)
        epsilon = 1e-6
        pred_prob = torch.clamp(pred_prob, epsilon, 1.0 - epsilon)
        
        # 4. Calculate Other Losses (Using Probabilities)
        loss_iou = self.iou(pred_prob, gt_f32) if self.weight_iou > 0 else 0.0
        loss_l1 = self.l1(pred_prob, gt_f32) if self.weight_l1 > 0 else 0.0
        loss_ssim = self.ssim(pred_prob, gt_f32) if self.weight_ssim > 0 else 0.0
        
        # 5. Feature Alignment Loss
        loss_feat = torch.tensor(0.0, device=pred_logits.device)
        if self.weight_feat > 0 and stu_feats is not None and tea_feats is not None:
            if isinstance(stu_feats, (list, tuple)):
                for s_f, t_f in zip(stu_feats, tea_feats):
                    loss_feat += self.mse(s_f.float(), t_f.float())
            else:
                loss_feat += self.mse(stu_feats.float(), tea_feats.float())
        
        # 6. Total Loss
        total_loss = (self.weight_bce * loss_bce) + \
                     (self.weight_iou * loss_iou) + \
                     (self.weight_ssim * loss_ssim) + \
                     (self.weight_l1 * loss_l1) + \
                     (self.weight_feat * loss_feat)
        
        # Detail metric
        loss_detail = loss_iou + loss_ssim
        
        return total_loss, loss_l1, loss_detail, loss_feat