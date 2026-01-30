# train.py

import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

# --- Import Custom Modules ---
from config import Config
from models.twinswinunet import TwinSwinUNet
from models.mask_encoder import MaskEncoder
from utils.dataset import MattingDataset
from utils.loss import TotalLoss
from utils.metrics import calculate_matting_metrics
from utils.logger import CSVLogger
from utils.plot import plot_history

def parse_args():
    parser = argparse.ArgumentParser(description="TwinSwin U-Net Training Script")
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    return parser.parse_args()

def train():
    args = parse_args()
    
    # =========================================================================
    # 1. Setup Environment & Paths
    # =========================================================================
    save_dir = Config.CHECKPOINT_DIR
    os.makedirs(save_dir, exist_ok=True)
    
    Config.print_info()
    print(f"📂 Output Directory: {save_dir}")

    device = torch.device(Config.DEVICE)
    torch.backends.cudnn.benchmark = True 

    # =========================================================================
    # 2. Data Loading
    # =========================================================================
    print("Dataset Loading...")
    train_root = os.path.join(Config.DATA_ROOT, Config.TRAIN_SET)
    val_root = os.path.join(Config.DATA_ROOT, Config.VAL_SET)

    train_dataset = MattingDataset(
        root_dir=train_root, mode='train', 
        target_size=Config.IMG_SIZE, schema=Config.SCHEMA      
    )
    val_dataset = MattingDataset(
        root_dir=val_root, mode='val', 
        target_size=Config.IMG_SIZE, schema=Config.SCHEMA
    )

    train_loader = DataLoader(
        train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY, drop_last=True 
    )
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        num_workers=Config.NUM_WORKERS, pin_memory=Config.PIN_MEMORY
    )
    
    print(f"   - Train Images: {len(train_dataset)} | Val Images: {len(val_dataset)}")

    # =========================================================================
    # 3. Model Initialization
    # =========================================================================
    model = TwinSwinUNet().to(device)
    
    teacher_model = None
    if Config.USE_TWIN_ALIGNMENT:
        print("🎓 Initializing Teacher Network (MaskEncoder)...")
        teacher_model = MaskEncoder(embed_dim=Config.EMBED_DIM).to(device)
        teacher_model.eval() 
        for p in teacher_model.parameters():
            p.requires_grad = False
    
    # =========================================================================
    # 4. Optimizer, Scheduler & Loss
    # =========================================================================
    optimizer = optim.AdamW(model.parameters(), lr=Config.LR, weight_decay=Config.WEIGHT_DECAY)
    
    # [Improvement] Added LR Scheduler for Swin Transformer stability
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=Config.NUM_EPOCHS)
    
    criterion = TotalLoss().to(device)
    scaler = GradScaler() 

    # =========================================================================
    # 5. History & Resume Logic
    # =========================================================================
    start_epoch = 0
    best_val_sad = float('inf') 
    history = {
        'train_loss': [], 'val_loss': [], 'train_sad': [], 'val_sad': [],
        'train_grad': [], 'val_grad': [], 'train_mse': [], 'val_mse': [],
        'train_acc': [], 'val_acc': []
    }

    checkpoint_path = os.path.join(save_dir, 'last_model.pth')
    if args.resume and os.path.exists(checkpoint_path):
        print(f"🔄 Resuming from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint['scaler'])
        if 'scheduler' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_sad = checkpoint.get('best_sad', float('inf'))
        history = checkpoint.get('history', history)
    
    logger = CSVLogger(save_dir=save_dir, filename='training_log.csv', resume=args.resume)

    # =========================================================================
    # 7. Training Loop
    # =========================================================================
    print(f"🚀 Starting Training (Epoch {start_epoch+1} -> {Config.NUM_EPOCHS})")
    accum_steps = Config.GRAD_ACCUM_STEPS

    for epoch in range(start_epoch, Config.NUM_EPOCHS):
        # --- TRAIN PHASE ---
        model.train()
        train_metrics = {'loss': 0, 'mse': 0, 'sad': 0, 'grad': 0, 'acc': 0}
        optimizer.zero_grad()

        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}/{Config.NUM_EPOCHS} [Train]")
        
        for step, (images, masks) in enumerate(pbar):
            # Dimension Safety Checks
            if images.dim() == 3: images = images.unsqueeze(0)
            if masks.dim() == 3: masks = masks.unsqueeze(0)   
            if masks.dim() == 3: masks = masks.unsqueeze(1)

            images, masks = images.to(device, non_blocking=True), masks.to(device, non_blocking=True)

            with autocast(device_type='cuda', enabled=Config.USE_AMP):
                outputs = model(images) 
                
                t_feats = None
                if teacher_model is not None:
                    with torch.no_grad():
                        masks_teacher = F.interpolate(masks, size=(Config.SAFE_SIZE, Config.SAFE_SIZE), mode='nearest')
                        t_feats = teacher_model(masks_teacher)

                loss, loss_dict = criterion(outputs, masks, teacher_feats=t_feats)
                loss = loss / accum_steps 

                # [FIX] Calculate Metrics properly during Training
                with torch.no_grad():
                    pred_prob = torch.sigmoid(outputs['fine']).detach()
                    mse, sad, grad, acc = calculate_matting_metrics(pred_prob, masks)
                    train_metrics['mse'] += mse
                    train_metrics['sad'] += sad
                    train_metrics['grad'] += grad
                    train_metrics['acc'] += acc

            scaler.scale(loss).backward()

            # Optimizer Step with Gradient Accumulation
            if (step + 1) % accum_steps == 0:
                # [Improvement] Gradient Clipping for stability
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            curr_l = loss.item() * accum_steps
            train_metrics['loss'] += curr_l
            pbar.set_postfix({'L': f"{curr_l:.4f}", 'SAD': f"{sad:.1f}"})

        # --- VALIDATION PHASE ---
        model.eval()
        val_metrics = {'loss': 0, 'mse': 0, 'sad': 0, 'grad': 0, 'acc': 0}
        pbar_v = tqdm(val_loader, desc=f"Ep {epoch+1} [Val]")
        
        with torch.no_grad():
            for imgs, msks in pbar_v:
                if imgs.dim() == 3: imgs = imgs.unsqueeze(0)
                if msks.dim() == 3: msks = msks.unsqueeze(0); msks = msks.unsqueeze(1)

                imgs, msks = imgs.to(device), msks.to(device)
                pred = model(imgs) # Eval mode returns Sigmoid output directly
                
                mse, sad, grad, acc = calculate_matting_metrics(pred, msks)
                val_metrics['loss'] += F.l1_loss(pred, msks).item()
                val_metrics['mse'] += mse; val_metrics['sad'] += sad; val_metrics['grad'] += grad; val_metrics['acc'] += acc

        # --- UPDATE STATISTICS ---
        # Helper to average metrics
        avg_t = {k: v / len(train_loader) for k, v in train_metrics.items()}
        avg_v = {k: v / len(val_loader) for k, v in val_metrics.items()}

        for k in history.keys():
            history[k].append(avg_t[k.split('_')[1]] if 'train' in k else avg_v[k.split('_')[1]])

        # Step the scheduler
        scheduler.step()

        # Log & Plot
        cur_lr = optimizer.param_groups[0]['lr']
        logger.log([epoch+1, cur_lr, avg_t['loss'], avg_t['mse'], avg_t['sad'], avg_t['grad'], avg_t['acc'],
                    avg_v['loss'], avg_v['mse'], avg_v['sad'], avg_v['grad'], avg_v['acc']])
        
        print(f"📊 Ep {epoch+1} Summary: Train SAD: {avg_t['sad']:.1f} | Val SAD: {avg_v['sad']:.1f} | Acc: {avg_v['acc']:.2f}%")
        plot_history(history['train_loss'], history['val_loss'], history['train_sad'], history['val_sad'], 
                     history['train_grad'], history['val_grad'], history['train_mse'], history['val_mse'], 
                     history['train_acc'], history['val_acc'], save_dir=save_dir)

        # Save Checkpoints
        state = {'epoch': epoch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 
                 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'best_sad': best_val_sad, 'history': history}
        torch.save(state, os.path.join(save_dir, 'last_model.pth'))

        if avg_v['sad'] < best_val_sad:
            best_val_sad = avg_v['sad']
            torch.save(state, os.path.join(save_dir, 'best_model.pth'))
            print(f"🏆 Saved New Best Model (SAD: {best_val_sad:.4f})")

if __name__ == '__main__':
    train()