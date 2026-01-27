# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Import Custom Modules
import config
from utils.dataset import DIS5KDataset
from models.twin_swin_unet import TwinSwinUNet
from models.mask_encoder import SwinMaskEncoder
from utils.loss import MattingLoss
from utils.logger import CSVLogger
from utils.metrics import calculate_matting_metrics
from utils.plot import plot_history

def train():
    # --- 1. Setup ---
    print(f"🚀 Starting training: {config.EXPERIMENT_NAME}")
    print(f"   Device: {config.DEVICE}")
    print(f"   Input Size: {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"   Dilation: {config.DILATE_MASK}")
    print(f"   Twin Alignment: {config.USE_TWIN_ALIGNMENT}")

    # Create directories
    logger = CSVLogger(config.LOG_DIR)
    
    # --- 2. Data Loading ---
    print("📂 Loading Datasets...")
    train_ds = DIS5KDataset(config.DATASET_ROOT, mode='train', 
                            target_size=config.IMG_SIZE, dilate_mask=config.DILATE_MASK)
    val_ds = DIS5KDataset(config.DATASET_ROOT, mode='val', 
                          target_size=config.IMG_SIZE, dilate_mask=False) # Val uses raw GT

    train_loader = DataLoader(train_ds, batch_size=config.BATCH_SIZE, 
                              shuffle=True, num_workers=config.NUM_WORKERS, 
                              pin_memory=config.PIN_MEMORY)
    val_loader = DataLoader(val_ds, batch_size=1, # Val batch size 1 for accurate metrics
                            shuffle=False, num_workers=config.NUM_WORKERS, 
                            pin_memory=config.PIN_MEMORY)

    print(f"   Train Images: {len(train_ds)}")
    print(f"   Val Images: {len(val_ds)}")

    # --- 3. Model Initialization ---
    # A. Student Model (The Locator)
    print(f"🔹 Initializing Student Model: {config.BACKBONE_NAME}")
    model = TwinSwinUNet(n_classes=1, img_size=config.IMG_SIZE, 
                         backbone_name=config.BACKBONE_NAME).to(config.DEVICE)
    
    # B. Teacher Model (Mask Encoder) - Only if alignment is enabled
    teacher = None
    if config.USE_TWIN_ALIGNMENT:
        # [FIX] Get embed_dim dynamically from the Student model
        # model.dims[0] corresponds to 'embed_dim' (e.g., 96 for Tiny, 128 for Base)
        student_embed_dim = model.dims[0]
        print(f"🎓 Initializing Teacher (Mask Encoder) with embed_dim={student_embed_dim}...")
        
        teacher = SwinMaskEncoder(embed_dim=student_embed_dim).to(config.DEVICE)
        
        teacher.eval() # Teacher is always in eval mode
        for param in teacher.parameters():
            param.requires_grad = False # Freeze Teacher

    # --- 4. Optimization ---
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2, eta_min=1e-6)
    
    # Use mixed precision scaler
    scaler = torch.amp.GradScaler('cuda') 

    loss_fn = MattingLoss(**config.LOSS_WEIGHTS).to(config.DEVICE)

    # --- 5. Training Loop ---
    best_iou = 0.0
    history = {'train_loss':[], 'val_loss':[], 'train_sad':[], 'val_sad':[], 
               'train_grad':[], 'val_grad':[], 'train_mse':[], 'val_mse':[], 'train_acc':[], 'val_acc':[]}

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_loss_epoch = 0
        train_metrics = [0, 0, 0, 0] # mse, sad, grad, acc

        train_loop = tqdm(train_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Train]")
        
        for images, masks in train_loop:
            images = images.to(config.DEVICE)
            masks = masks.to(config.DEVICE)

            # --- Forward ---
            with torch.amp.autocast('cuda'):
                # 1. Teacher Forward (Get Target Features)
                tea_feats = None
                if teacher is not None:
                    with torch.no_grad():
                        tea_feats = teacher(masks)

                # 2. Student Forward
                output = model(images)
                
                stu_feats = None
                if isinstance(output, tuple):
                    preds, stu_feats = output
                else:
                    preds = output

                # 3. Calculate Loss
                loss, _, _, _ = loss_fn(preds, masks, stu_feats, tea_feats)

            # --- Backward ---
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # --- Metrics ---
            train_loss_epoch += loss.item()
            with torch.no_grad():
                # Sigmoid for metric calculation (preds are logits)
                pred_final = torch.sigmoid(preds)
                m_vals = calculate_matting_metrics(pred_final, masks)
                for i in range(4): train_metrics[i] += m_vals[i]

            train_loop.set_postfix(loss=loss.item())

        # End of Epoch Scheduling
        scheduler.step()

        # Averages
        train_loss_epoch /= len(train_loader)
        train_metrics = [x / len(train_loader) for x in train_metrics]

        # --- Validation ---
        val_loss_epoch = 0
        val_metrics = [0, 0, 0, 0]
        model.eval()

        val_loop = tqdm(val_loader, desc=f"Epoch {epoch}/{config.NUM_EPOCHS} [Val]")

        with torch.no_grad():
            for images, masks in val_loop:
                images = images.to(config.DEVICE)
                masks = masks.to(config.DEVICE)

                # Validation Forward (No Teacher needed)
                output = model(images)
                if isinstance(output, tuple): preds, _ = output
                else: preds = output
                
                # Validation Loss (Alignment skipped)
                # Note: We need to use autocast here too if using AMP models
                with torch.amp.autocast('cuda'):
                    loss, _, _, _ = loss_fn(preds, masks) 
                
                val_loss_epoch += loss.item()
                
                pred_final = torch.sigmoid(preds)
                m_vals = calculate_matting_metrics(pred_final, masks)
                for i in range(4): val_metrics[i] += m_vals[i]

                val_loop.set_postfix(loss=loss.item())

        val_loss_epoch /= len(val_loader)
        val_metrics = [x / len(val_loader) for x in val_metrics]

        # --- Logging ---
        log_data = [epoch, optimizer.param_groups[0]['lr'], 
                    train_loss_epoch, *train_metrics, 
                    val_loss_epoch, *val_metrics]
        logger.log(log_data)

        # Update History
        history['train_loss'].append(train_loss_epoch); history['val_loss'].append(val_loss_epoch)
        history['train_mse'].append(train_metrics[0]); history['val_mse'].append(val_metrics[0])
        history['train_sad'].append(train_metrics[1]); history['val_sad'].append(val_metrics[1])
        history['train_grad'].append(train_metrics[2]); history['val_grad'].append(val_metrics[2])
        history['train_acc'].append(train_metrics[3]); history['val_acc'].append(val_metrics[3])

        print(f"📉 Epoch {epoch} | Train Loss: {train_loss_epoch:.4f} | Val Loss: {val_loss_epoch:.4f} | Val IoU(Acc): {val_metrics[3]:.2f}%")

        # --- Save Best Model ---
        current_iou = val_metrics[3]
        if current_iou > best_iou:
            best_iou = current_iou
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            print(f"💾 Best Model Saved! IoU: {best_iou:.2f}%")
        
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)

        # Plot every 5 epochs
        if epoch % 5 == 0:
            plot_history(history['train_loss'], history['val_loss'], 
                         history['train_sad'], history['val_sad'],
                         history['train_grad'], history['val_grad'],
                         history['train_mse'], history['val_mse'],
                         history['train_acc'], history['val_acc'],
                         config.LOG_DIR)

if __name__ == "__main__":
    train()