# utils/metrics.py

import torch
import torch.nn.functional as F

def calculate_matting_metrics(pred_alpha, gt_alpha):
    """
    Calculate Matting Metrics: MSE and Pixel Accuracy.
    
    Args:
        pred_alpha (Tensor): (N, 1, H, W), range [0, 1]
        gt_alpha (Tensor): (N, 1, H, W), range [0, 1]
        
    Returns:
        mse (float): Mean Squared Error (Lower is better)
        acc (float): Pixel-wise Accuracy (Higher is better, 0~100%)
    """
    # Detach and ensure float
    pred = pred_alpha.detach().float()
    gt = gt_alpha.detach().float()
    
    # 1. MSE (Mean Squared Error)
    mse = F.mse_loss(pred, gt).item()
    
    # 2. Accuracy (Derived from MAD)
    # MAD = Mean Absolute Difference
    abs_diff = torch.abs(pred - gt)
    mad = abs_diff.mean().item()
    
    # Accuracy = (1 - Error) * 100
    acc = (1.0 - mad) * 100.0
    
    return mse, acc