import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

# --- Import Custom Modules ---
import config
from models.twin_swin_matte import TwinSwinMatteNet
from utils.dataset import MattingDataset
from utils.loss import MattingLoss
from utils.metrics import calculate_matting_metrics
from utils.logger import CSVLogger
from utils.plot import plot_history

def get_dataloaders():
    """
    建立訓練與驗證的 DataLoader
    """
    print(f"📂 Dataset Root: {config.DATASET_ROOT}")
    
    train_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='train',
        img_size=config.IMG_SIZE
    )
    
    val_ds = MattingDataset(
        root_dir=config.DATASET_ROOT,
        mode='val',
        img_size=config.IMG_SIZE
    )

    train_loader = DataLoader(
        train_ds, 
        batch_size=config.BATCH_SIZE, 
        shuffle=True, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    # 驗證集 Batch Size 建議設為 1 或較小，以避免 OOM (如果圖片很大)
    val_loader = DataLoader(
        val_ds, 
        batch_size=max(1, config.BATCH_SIZE // 2), 
        shuffle=False, 
        num_workers=config.NUM_WORKERS, 
        pin_memory=config.PIN_MEMORY
    )
    
    return train_loader, val_loader

def build_model_and_optimizer(device):
    """
    建立模型、優化器、Loss 函數
    """
    print(f"🏗️ Building Model: TwinSwinMatteNet ({config.BACKBONE})...")
    
    model = TwinSwinMatteNet(
        n_classes=config.NUM_CLASSES,
        img_size=config.IMG_SIZE,
        backbone_name=config.BACKBONE,
        pretrained=True
    ).to(device)

    # --- Optimizer 設定 ---
    # 策略: Backbone 使用較小的 LR，Decoder 使用較大的 LR
    # 重要: 必須排除 gt_encoder (Teacher) 的參數
    
    # 1. 取得 img_encoder (Student Backbone) 的參數 ID
    img_enc_ids = list(map(id, model.img_encoder.parameters()))
    
    # 2. 取得 gt_encoder (Teacher Backbone) 的參數 ID (這些不該被訓練)
    gt_enc_ids = list(map(id, model.gt_encoder.parameters()))
    
    # 3. 篩選出 Decoder 和其他部分的參數
    decoder_params = filter(lambda p: id(p) not in img_enc_ids and id(p) not in gt_enc_ids, model.parameters())

    optimizer = optim.AdamW([
        {'params': decoder_params, 'lr': config.LEARNING_RATE},           # Decoder: 正常 LR
        {'params': model.img_encoder.parameters(), 'lr': config.LEARNING_RATE * 0.1} # Backbone: 0.1x LR
    ], weight_decay=1e-3)

    # --- Loss Function ---
    # weight_feat=0.5 表示啟動 Feature Consistency Loss
    criterion = MattingLoss(
        weight_bce=1.0, weight_l1=1.0, weight_ssim=0.5, weight_iou=0.5, weight_feat=0.5
    ).to(device)
    
    # --- Scheduler ---
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=config.SCHEDULER_T0, 
        T_mult=config.SCHEDULER_T_MULT, 
        eta_min=config.SCHEDULER_ETA_MIN
    )
    
    return model, optimizer, criterion, scheduler

def train_one_epoch(loader, model, optimizer, criterion, scaler, device, epoch):
    model.train()
    loop = tqdm(loader, desc=f"Train Ep {epoch}", leave=True)
    
    avg_loss = 0
    avg_mse = 0
    avg_acc = 0
    
    for batch_idx, (images, masks) in enumerate(loop):
        images = images.to(device)
        masks = masks.to(device)
        
        optimizer.zero_grad()
        
        # --- Mixed Precision Forward ---
        with torch.amp.autocast('cuda'):
            # 重要: 訓練時傳入 gt_mask，讓 Teacher (gt_encoder) 產生特徵
            pred_alpha, stu_feats, tea_feats = model(images, gt_mask=masks)
            
            # 計算 Loss (包含 Feature Consistency)
            loss, loss_l1, loss_detail, loss_feat = criterion(pred_alpha, masks, stu_feats, tea_feats)

        # --- Backward ---
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        # --- Metrics & Logging ---
        mse, acc = calculate_matting_metrics(pred_alpha, masks)
        
        # 更新平均值
        avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
        avg_mse = (avg_mse * batch_idx + mse) / (batch_idx + 1)
        avg_acc = (avg_acc * batch_idx + acc) / (batch_idx + 1)
        
        loop.set_postfix(loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.4f}", acc=f"{avg_acc:.2f}%")
        
    return avg_loss, avg_mse, avg_acc

def validate(loader, model, criterion, device, epoch):
    model.eval()
    loop = tqdm(loader, desc=f"Val Ep {epoch}", leave=True)
    
    avg_loss = 0
    avg_mse = 0
    avg_acc = 0
    
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(loop):
            images = images.to(device)
            masks = masks.to(device)
            
            # --- Inference Forward ---
            # 重要: 驗證時 gt_mask=None，Teacher 不運作
            pred_alpha, stu_feats, _ = model(images, gt_mask=None)
            
            # 計算 Loss (此時 tea_feats=None，MattingLoss 會自動忽略 Feature Loss)
            loss, _, _, _ = criterion(pred_alpha, masks, stu_feats, None)
            
            # Metrics
            mse, acc = calculate_matting_metrics(pred_alpha, masks)
            
            avg_loss = (avg_loss * batch_idx + loss.item()) / (batch_idx + 1)
            avg_mse = (avg_mse * batch_idx + mse) / (batch_idx + 1)
            avg_acc = (avg_acc * batch_idx + acc) / (batch_idx + 1)
            
            loop.set_postfix(loss=f"{avg_loss:.4f}", mse=f"{avg_mse:.4f}", acc=f"{avg_acc:.2f}%")
            
    return avg_loss, avg_mse, avg_acc

def main():
    # 1. 確保目錄存在
    if not os.path.exists(config.SAVE_DIR):
        os.makedirs(config.SAVE_DIR)
        print(f"📁 Created Checkpoint Dir: {config.SAVE_DIR}")

    # 2. 準備組件
    train_loader, val_loader = get_dataloaders()
    model, optimizer, criterion, scheduler = build_model_and_optimizer(config.DEVICE)
    scaler = torch.amp.GradScaler('cuda')
    logger = CSVLogger(config.SAVE_DIR, filename='training_log.csv')
    
    best_acc = 0.0
    history = {
        'train_loss': [], 'val_loss': [],
        'train_mse': [], 'val_mse': [],
        'train_acc': [], 'val_acc': []
    }

    print(f"\n🚀 Start Training: {config.EXPERIMENT_NAME}")
    print(f"   Epochs: {config.NUM_EPOCHS} | Batch: {config.BATCH_SIZE} | Img Size: {config.IMG_SIZE}")
    print("-" * 60)

    # 3. 訓練迴圈
    for epoch in range(1, config.NUM_EPOCHS + 1):
        print(f"\nExample Epoch {epoch}/{config.NUM_EPOCHS}")
        
        # Train
        t_loss, t_mse, t_acc = train_one_epoch(train_loader, model, optimizer, criterion, scaler, config.DEVICE, epoch)
        
        # Validate
        v_loss, v_mse, v_acc = validate(val_loader, model, criterion, config.DEVICE, epoch)
        
        # Scheduler Step
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log Data
        logger.log([epoch, current_lr, t_loss, t_mse, t_acc, v_loss, v_mse, v_acc])
        
        # Save History for Plotting
        history['train_loss'].append(t_loss); history['val_loss'].append(v_loss)
        history['train_mse'].append(t_mse);   history['val_mse'].append(v_mse)
        history['train_acc'].append(t_acc);   history['val_acc'].append(v_acc)
        
        # Save Models
        torch.save(model.state_dict(), config.LAST_MODEL_PATH)
        
        if v_acc > best_acc:
            best_acc = v_acc
            print(f"⭐ New Best Accuracy: {best_acc:.2f}% (Saved)")
            torch.save(model.state_dict(), config.BEST_MODEL_PATH)
            
        # Plot Curves
        plot_history(
            history['train_loss'], history['val_loss'],
            history['train_mse'], history['val_mse'],
            history['train_acc'], history['val_acc'],
            config.SAVE_DIR
        )

    print("\n✅ Training Completed!")

if __name__ == "__main__":
    main()