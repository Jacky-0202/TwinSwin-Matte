# config.py

import os
import torch
from datetime import datetime

class Config:
    # =========================================================================
    # 1. Core Task Settings (Switch Task Here!)
    # =========================================================================
    # Options: 'DIS5K', 'COD10K', 'HRS10K', 'HRSOD'
    TASK_NAME = 'DIS5K'  

    # Task Mode (Determines Loss Function Strategy)
    # 'MATTING': Focus on fine details (L1, Grad, SSIM) -> Best for DIS5K, HRS10K, HRSOD
    # 'LOCATOR': Focus on positioning (BCE, IoU/Dice)   -> Best for COD10K (Camouflaged Object)
    MODE = 'MATTING' 

    # =========================================================================
    # 2. Dataset Path Configuration
    # =========================================================================
    # Base root directory for all datasets
    BASE_ROOT = "/home/tec/Desktop/Project/Datasets/Matte"

    # Auto-configure paths based on TASK_NAME
    if TASK_NAME == 'DIS5K':
        DATA_ROOT = os.path.join(BASE_ROOT, 'DIS5K')
        TRAIN_SET = 'DIS-TR'    # Tree: DIS5K/DIS-TR/im
        VAL_SET   = 'DIS-VD'    # Tree: DIS5K/DIS-VD/im
        TEST_SET  = 'DIS-TE1'   
        SCHEMA    = 'standard'  # Standard 'im'/'gt' structure

    elif TASK_NAME == 'HRS10K':
        DATA_ROOT = os.path.join(BASE_ROOT, 'HRS10K')
        TRAIN_SET = 'TR-HRS10K' # Tree: HRS10K/TR-HRS10K/im
        VAL_SET   = 'TE-HRS10K' # Using Test set as Val
        TEST_SET  = 'TE-HRS10K'
        SCHEMA    = 'standard'

    elif TASK_NAME == 'HRSOD':
        DATA_ROOT = os.path.join(BASE_ROOT, 'HRSOD')
        TRAIN_SET = 'TR-HRSOD'  # Tree: HRSOD/TR-HRSOD/im
        VAL_SET   = 'HRSOD_TEsets/TE-HRSOD' 
        TEST_SET  = 'HRSOD_TEsets/TE-HRSOD'
        SCHEMA    = 'standard'

    elif TASK_NAME == 'COD10K':
        DATA_ROOT = os.path.join(BASE_ROOT, 'COD10K')
        TRAIN_SET = 'COD_TRsets/TR-COD10K' 
        VAL_SET   = 'COD_TEsets/TE-COD10K'
        TEST_SET  = 'COD_TEsets/TE-COD10K'
        # 'cod10k' schema handles 'Train/Image' or 'Imgs/GT' variations
        SCHEMA    = 'cod10k' 

    elif TASK_NAME == 'Custom':
        # Perfect for your own collected data or quick experiments.
        DATA_ROOT = os.path.join(BASE_ROOT, 'Custom')
        
        # Structure expectation:
        #   Custom/train/im, Custom/train/gt
        #   Custom/val/im,   Custom/val/gt
        TRAIN_SET = 'train'
        VAL_SET   = 'val'
        TEST_SET  = 'val' # Fallback to val if no test set
        
        # Use 'standard' (im/gt) or 'cod10k' (Image/GT) depending on your habit
        SCHEMA    = 'standard'

    # =========================================================================
    # 3. Model Architecture & Hardware Settings
    # =========================================================================
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # [Model Selection]
    # Options: 'tiny', 'small', 'base', 'base384', 'large'
    MODEL_TYPE = 'base'

    # [Backbone Zoo Configuration]
    # 'dim': Input channels for Teacher MaskEncoder (Base=128, Large=192)
    # 'safe_size': Optimal resolution to prevent Swin Window Partition errors
    _BACKBONE_CONFIG = {
        'tiny': {
            'name': 'swin_tiny_patch4_window7_224', 
            'dim': 96, 
            'safe_size': 896 
        },
        'small': {
            'name': 'swin_small_patch4_window7_224', 
            'dim': 96, 
            'safe_size': 896 
        },
        'base': {
            'name': 'swin_base_patch4_window7_224', 
            'dim': 128, 
            'safe_size': 896 
        },
        'base384': {
            'name': 'swin_base_patch4_window12_384', 
            'dim': 128, 
            'safe_size': 1152 # 384 * 3
        },
        'large': {
            'name': 'swin_large_patch4_window12_384_in22k', 
            'dim': 192, 
            'safe_size': 1152 # 384 * 3
        },
    }

    # Validate and Load Model Parameters
    if MODEL_TYPE not in _BACKBONE_CONFIG:
        raise ValueError(f"❌ Unknown MODEL_TYPE: {MODEL_TYPE}. Options: {list(_BACKBONE_CONFIG.keys())}")

    _cfg = _BACKBONE_CONFIG[MODEL_TYPE]
    BACKBONE_NAME = _cfg['name']
    EMBED_DIM     = _cfg['dim']       # For Teacher Network
    SAFE_SIZE     = _cfg['safe_size'] # For Decoupled Strategy
    
    # [H200 Resolution Strategy]
    IMG_SIZE = 1024             # Final Target Resolution
    RESOLUTION_DECOUPLED = True # Resize to SAFE_SIZE for backbone, then restore

    # [Twin Alignment Strategy]
    USE_TWIN_ALIGNMENT = False # Enable Teacher-Student Feature Alignment
    
    # [Data Augmentation]
    DILATE_MASK = True        # Slightly dilate mask (Helps robustness)

    # [Memory & Speed]
    BATCH_SIZE = 16            # Recommended: 2 for Large Model
    NUM_WORKERS = 4            # CPU cores
    PIN_MEMORY = True         
    USE_AMP = True            # Mixed Precision (Essential for H200)
    GRAD_ACCUM_STEPS = 4      # Effective Batch Size = 2 * 4 = 8

    # =========================================================================
    # 4. Training Hyperparameters
    # =========================================================================
    NUM_EPOCHS = 70          
    LR = 1e-4                 # Initial Learning Rate
    WEIGHT_DECAY = 1e-4       
    
    # =========================================================================
    # 5. Loss Function Auto-Weighting
    # =========================================================================
    if MODE == 'MATTING':
        LOSS_WEIGHTS = {
            'bce': 0.0,
            'dice': 0.0,
            'l1': 1.0,        # Pixel-level L1 Loss
            'grad': 1.0,      # Gradient Loss (Sharpness)
            'ssim': 0.5,      # Structural Similarity
            'feat': 0.2       # Feature Alignment Loss
        }
    elif MODE == 'LOCATOR':
        LOSS_WEIGHTS = {
            'bce': 1.0,       # Structure Loss (BiRefNet style)
            'dice': 1.0,      # IoU Optimization
            'l1': 0.0,
            'grad': 0.0,
            'ssim': 0.0,
            'feat': 0.1
        }

    # =========================================================================
    # 6. Output Paths
    # =========================================================================
    # Checkpoints will be saved here
    timestamp = datetime.now().strftime("%Y%m%d%H%M")
    CHECKPOINT_DIR = f"./checkpoints/TwinSwin_{TASK_NAME}_{MODE}_{IMG_SIZE}_{timestamp}"

    @staticmethod
    def print_info():
        print("\n" + "="*40)
        print(f"🚀 CONFIGURATION: {Config.TASK_NAME}")
        print(f"   Mode:        {Config.MODE}")
        print(f"   Root:        {Config.DATA_ROOT}")
        print(f"   Train Set:   {Config.TRAIN_SET}")
        print("-" * 40)
        print(f"   Model:       {Config.MODEL_TYPE} ({Config.BACKBONE_NAME})")
        print(f"   Resolution:  {Config.IMG_SIZE} (Safe: {Config.SAFE_SIZE})")
        print(f"   Batch Size:  {Config.BATCH_SIZE} (Accum: {Config.GRAD_ACCUM_STEPS})")
        print(f"   Weights:     {Config.LOSS_WEIGHTS}")
        print("="*40 + "\n")