# utils/loss.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class MattingLoss(nn.Module):
    """
    Composite Loss for High-Resolution Matting (TwinSwin-Matte).
    
    Components:
    1. Alpha L1 Loss: Absolute error between prediction and GT.
    2. Gradient Loss (Sobel): Ensures sharp edges and details (hair).
    3. Feature Loss: MSE between Student (Adapted) and Teacher features.
    """
    def __init__(self, weight_alpha=1.0, weight_grad=1.0, weight_feat=0.5):
        super(MattingLoss, self).__init__()
        self.weight_alpha = weight_alpha
        self.weight_grad = weight_grad
        self.weight_feat = weight_feat
        
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
        
        # Define Sobel Kernels for Gradient Loss
        # shape: (out_channels, in_channels/groups, k, k)
        self.sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3)
        self.sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1, 1, 3, 3)

    def compute_gradient_loss(self, pred, gt):
        """
        Computes L1 loss on image gradients (edges).
        """
        # Ensure kernels are on the same device
        if self.sobel_x.device != pred.device:
            self.sobel_x = self.sobel_x.to(pred.device)
            self.sobel_y = self.sobel_y.to(pred.device)
            
        # Compute Gradients for Prediction
        pred_grad_x = F.conv2d(pred, self.sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, self.sobel_y, padding=1)
        
        # Compute Gradients for GT
        gt_grad_x = F.conv2d(gt, self.sobel_x, padding=1)
        gt_grad_y = F.conv2d(gt, self.sobel_y, padding=1)
        
        # Calculate L1 error on gradients
        loss_dx = F.l1_loss(torch.abs(pred_grad_x), torch.abs(gt_grad_x))
        loss_dy = F.l1_loss(torch.abs(pred_grad_y), torch.abs(gt_grad_y))
        
        return loss_dx + loss_dy

    def forward(self, pred_alpha, gt_alpha, stu_feats=None, tea_feats=None):
        """
        Args:
            pred_alpha: Predicted Alpha Matte (N, 1, H, W), range 0~1
            gt_alpha: Ground Truth Alpha (N, 1, H, W), range 0~1
            stu_feats: Tuple of Student features (Stage2, Stage0)
            tea_feats: Tuple of Teacher features (Stage2, Stage0)
        """
        # 1. Alpha L1 Loss (Pixel-level accuracy)
        loss_alpha = self.l1(pred_alpha, gt_alpha)
        
        # 2. Gradient Loss (Edge sharpness)
        loss_grad = self.compute_gradient_loss(pred_alpha, gt_alpha)
        
        # 3. Feature Consistency Loss (TwinSwin Core)
        # Only calculate if Teacher features are provided (Training phase)
        loss_feat = torch.tensor(0.0, device=pred_alpha.device)
        
        if stu_feats is not None and tea_feats is not None:
            # stu_feats = (c3_adapt, c1_adapt)
            # tea_feats = (teacher_c3, teacher_c1)
            
            # High-level Structure Loss (Stage 2)
            loss_feat_high = self.mse(stu_feats[0], tea_feats[0])
            
            # Low-level Detail Loss (Stage 0)
            loss_feat_low = self.mse(stu_feats[1], tea_feats[1])
            
            loss_feat = loss_feat_high + loss_feat_low

        # Combine all losses
        total_loss = (self.weight_alpha * loss_alpha) + \
                     (self.weight_grad * loss_grad) + \
                     (self.weight_feat * loss_feat)
                     
        return total_loss, loss_alpha, loss_grad, loss_feat