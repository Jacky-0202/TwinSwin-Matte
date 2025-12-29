# train.py

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm
import math

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

# --- Imports ---
import config as config
# [CHANGED] Import TwinSwinMatteNet instead of SwinASPPNet
from models.twin_swin_matte import TwinSwinMatteNet 
from utils.logger import CSVLogger
from utils.plot import plot_history
# [CHANGED] Import Matting-specific Loss and Dataset
from utils.loss import MattingLoss
from utils.dataset import MattingDataset
# [CHANGED] Import SAD/MSE calculation
from utils.metrics import calculate_matting_metrics

# --- 3. Setup Functions ---
def get_loaders():
    """Initializes and returns Train and Validation DataLoaders for Matting."""
    print(f"📂 Dataset Root: {config.DATASET_ROOT}")
    
    # Initialize Train Dataset (DIS5K/P3M)
    train_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='train',
        img_size=config.IMG_SIZE # e.g., 768
    )
    
    # Initialize Val Dataset
    val_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='val',
        img_size=config.IMG_SIZE # This size is used for initial resize, inference might differ
    )

    train_loader = DataLoader(
        train_ds, batch_size=config.BATCH_SIZE, shuffle=True, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    # Batch Size can be 1 for validation to handle high-res inference
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False, 
        num_workers=config.NUM_WORKERS, pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader

def get_model_components(device):
    """
    Initializes TwinSwinMatteNet, MattingLoss, Optimizer, and Scheduler.
    """
    print(f"🏗️ Building Model: TwinSwinMatteNet ({config.BACKBONE})...")
    
    # Initialize the TwinSwin Model
    model = TwinSwinMatteNet(
        n_classes=config.NUM_CLASSES, # Should be 1 for Alpha
        img_size=config.IMG_SIZE,
        backbone_name=config.BACKBONE,
        pretrained=True
    ).to(device)
    
    # Initialize Composite Loss (L1 + Laplacian + Feature Consistency)
    loss_fn = MattingLoss()
    
    # --- Optimizer Setup (Exclude gt_encoder from Optimization) ---
    
    # 1. Identify img_encoder Backbone Parameters
    img_encoder_ids = list(map(id, model.img_encoder.parameters()))
    
    # 2. Identify gt_encoder Parameters (Should be ignored anyway, but good for safety)
    gt_encoder_ids = list(map(id, model.gt_encoder.parameters()))
    
    # 3. Identify Head/Decoder Parameters (Everything not in img_encoder or gt_encoder)
    # This includes Decoder, RFB, and Adapters
    base_params = filter(lambda p: id(p) not in img_encoder_ids and id(p) not in gt_encoder_ids, model.parameters())

    optimizer = optim.AdamW([
        # Group 1: Decoder/Neck/Adapters -> Full Learning Rate
        {'params': base_params, 'lr': config.LEARNING_RATE},
        
        # Group 2: img_encoder Backbone -> 0.1x Learning Rate
        {'params': model.img_encoder.parameters(), 'lr': config.LEARNING_RATE * 0.1}
    ], weight_decay=1e-2)
    
    # Mixed Precision Scaler
    scaler = torch.amp.GradScaler()
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.SCHEDULER_T0, 
        T_mult=config.SCHEDULER_T_MULT, 
        eta_min=config.SCHEDULER_ETA_MIN
    )
    
    return model, loss_fn, optimizer, scaler, scheduler

# --- 4. Core Loops ---
def run_epoch(loader, model, optimizer, loss_fn, scaler, device, mode='train'):
    """
    Runs one epoch of training or validation for Matting task.
    """
    model.train() if mode == 'train' else model.eval()
    loop = tqdm(loader, desc=mode.capitalize(), leave=False)
    
    total_loss = 0.0
    total_mse = 0.0
    total_acc = 0.0
    
    # Track sub-losses for debugging
    avg_alpha_loss = 0.0
    avg_lap_loss = 0.0
    avg_feat_loss = 0.0
    
    with torch.set_grad_enabled(mode == 'train'):
        for data, targets in loop:
            data, targets = data.to(device), targets.to(device)

            if mode == 'train':
                # --- Training Phase ---
                with torch.amp.autocast('cuda'):
                    # Forward pass with GT Mask (for gt_encoder)
                    # Returns: pred, (img_feats), (gt_feats)
                    predictions, img_feats, gt_feats = model(data, gt_mask=targets)
                    
                    # Calculate Composite Loss
                    loss, l_alpha, l_lap, l_feat = loss_fn(predictions, targets, img_feats, gt_feats)

                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate sub-losses for logging
                avg_alpha_loss += l_alpha.item()
                avg_lap_loss += l_lap.item()
                avg_feat_loss += l_feat.item()

            else:
                # --- Validation Phase ---
                # 1. Resize for Inference (Use IMG_SIZE from config)
                orig_h, orig_w = data.shape[-2:]
                
                # Resize input to IMG_SIZE (e.g., 1024x1024)
                data_resized = F.interpolate(
                    data, size=config.IMG_SIZE, mode='bilinear', align_corners=True
                )
                
                with torch.amp.autocast('cuda'):
                    # Forward pass WITHOUT GT Mask (gt_encoder is idle)
                    # Returns: pred, img_feats, None
                    logits, _, _ = model(data_resized, gt_mask=None)
                    
                    # Upsample prediction back to original size
                    predictions = F.interpolate(
                        logits, size=(orig_h, orig_w), mode='bilinear', align_corners=True
                    )
                    
                    # Compute Validation Loss (Only Alpha + Laplacian, no Feature loss)
                    loss, l_alpha, l_lap, _ = loss_fn(predictions, targets, None, None)

            # --- Metrics Calculation (MSE) ---
            mse_val, acc_val = calculate_matting_metrics(predictions, targets)
            
            total_loss += loss.item()
            total_mse += mse_val
            total_acc += acc_val
            
            # Update progress bar
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc_val:.2f}%")
            
    # Calculate Averages
    num_batches = len(loader)
    
    return (
        total_loss / num_batches, 
        total_mse / num_batches,
        total_acc / num_batches
    )

# --- 5. Main Execution ---
def main():
    print(f"--- TwinSwin-Matte Training Setup ---")
    
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(config.SAVE_DIR, exist_ok=True)
    
    logger = CSVLogger(save_dir=config.SAVE_DIR, filename='training_log.csv')
    
    train_loader, val_loader = get_loaders()
    model, loss_fn, optimizer, scaler, scheduler = get_model_components(config.DEVICE)

    best_acc = 0.0
    
    history = {'train_loss': [], 'val_loss': [], 
               'train_mse': [], 'val_mse': [],
               'train_acc': [], 'val_acc': []}

    print(f"\n🚀 Starting Training for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch [{epoch+1}/{config.NUM_EPOCHS}] LR: {current_lr:.6f}")

        # Train (Expects 3 values)
        train_loss, train_mse, train_acc = run_epoch(
            train_loader, model, optimizer, loss_fn, scaler, config.DEVICE, mode='train'
        )
        
        # Val (Expects 3 values)
        val_loss, val_mse, val_acc = run_epoch(
            val_loader, model, None, loss_fn, None, config.DEVICE, mode='val'
        )
        
        scheduler.step()
        
        # Log to CSV
        logger.log([epoch+1, current_lr, train_loss, train_mse, train_acc, val_loss, val_mse, val_acc])
        
        print(f"\tTrain Loss: {train_loss:.4f} | MSE: {train_mse:.5f} | Acc: {train_acc:.2f}%")
        print(f"\tVal Loss:   {val_loss:.4f} | MSE: {val_mse:.5f} | Acc: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss); history['val_loss'].append(val_loss)
        history['train_mse'].append(train_mse);   history['val_mse'].append(val_mse)
        history['train_acc'].append(train_acc);   history['val_acc'].append(val_acc)

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! (Acc: {best_acc:.2f}%)")
            
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

    print("\n🎉 Training Complete!")
    
    plot_history(
        history['train_loss'], history['val_loss'], 
        history['train_mse'], history['val_mse'], 
        history['train_acc'], history['val_acc'], 
        save_dir=config.SAVE_DIR
    )

if __name__ == "__main__":
    main()