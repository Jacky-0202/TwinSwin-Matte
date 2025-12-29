# config.py

import torch
import os

# --- 1. Device Setup ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. Model Selection (List Mode) ---
# List of available Swin Transformer backbones supported by timm.
# Note: 384 versions provide better segmentation results but require more VRAM.
SWIN_VARIANTS = [
    # --- 224x224 Input (Window Size 7) ---
    'swin_tiny_patch4_window7_224',    # [0] Tiny  (96 dim)  - Fastest, Low VRAM
    'swin_small_patch4_window7_224',   # [1] Small (96 dim)  - Balanced
    'swin_base_patch4_window7_224',    # [2] Base  (128 dim) - Strong
    
    # --- 384x384 Input (Window Size 12) ---
    'swin_base_patch4_window12_384',   # [3] Base  (128 dim) - High Res, Heavy VRAM
    'swin_large_patch4_window12_384'   # [4] Large (192 dim) - SOTA Level, Very Heavy
]

# 👉 Change this index to switch models!
# 0=Tiny(224), 1=Small(224), 2=Base(224), 3=Base(384), 4=Large(384)
MODEL_IDX = 1

# Select backbone safely
try:
    BACKBONE = SWIN_VARIANTS[MODEL_IDX]
except IndexError:
    print(f"⚠️ Invalid MODEL_IDX: {MODEL_IDX}, defaulting to [0] Tiny")
    BACKBONE = SWIN_VARIANTS[0]

# --- 3. Hyperparameters ---
IMG_SIZE = 512        # Image Size
BATCH_SIZE = 8         # Adjust based on VRAM (e.g., 16 for Tiny, 4-8 for Base/384)
NUM_CLASSES = 1        # Number of classes
NUM_EPOCHS = 2       # Total training epochs
NUM_WORKERS = 4        # Number of data loading threads
PIN_MEMORY = True      # Accelerate data transfer to GPU

LEARNING_RATE = 1e-4   # Transformers typically require lower LR than CNNs
SCHEDULER_T0 = 10
SCHEDULER_T_MULT = 2
SCHEDULER_ETA_MIN = 1e-6

# --- 4. Paths ---
# Adjust these paths according to your environment
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Dataset Root Directory
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/DIS5K_Flat"

# Expected DIS5K folder structure:
# DIS5K_Flat/
#   ├── train/
#   │   ├── im/  (Images)
#   │   └── gt/  (Masks)
#   └── val/ ...

TRAIN_IMG_DIR = os.path.join(DATASET_ROOT, 'train/im')
TRAIN_MASK_DIR = os.path.join(DATASET_ROOT, 'train/gt')
VAL_IMG_DIR = os.path.join(DATASET_ROOT, 'val/im')
VAL_MASK_DIR = os.path.join(DATASET_ROOT, 'val/gt')

# --- 5. Checkpoints & Logging ---
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints')

# Generate a unique experiment name based on the selected model
# e.g., "Swin_Tiny_ADE20K" or "Swin_Base384_ADE20K"
model_tag = 'Unknown'
if 'tiny' in BACKBONE: model_tag = 'Tiny'
elif 'small' in BACKBONE: model_tag = 'Small'
elif 'base' in BACKBONE: model_tag = 'Base'
elif 'large' in BACKBONE: model_tag = 'Large'

if '384' in BACKBONE:
    model_tag += '_384'

# EXPERIMENT_NAME = f"Swin_{model_tag}_ADE20K"
EXPERIMENT_NAME = f"TwinSwin_{model_tag}_DIS5K_test"
SAVE_DIR = os.path.join(CHECKPOINT_DIR, EXPERIMENT_NAME)

BEST_MODEL_PATH = os.path.join(SAVE_DIR, 'best_model.pth')
LAST_MODEL_PATH = os.path.join(SAVE_DIR, 'last_model.pth')

# Ensure the save directory exists
os.makedirs(SAVE_DIR, exist_ok=True)
