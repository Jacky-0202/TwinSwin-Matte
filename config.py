# config.py

import torch
import os

# --- 1. Device & System ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4         # Number of data loading threads
PIN_MEMORY = True       # Accelerate data transfer to GPU

# --- 2. Model Selection (Easy Switch) ---
# Dictionary of available Swin Transformer backbones
# Note: 'window7' models are native 224x224, 'window12' are native 384x384.
# TimM will handle resizing positional embeddings automatically.
BACKBONE_OPTIONS = {
    'tiny':  'swin_tiny_patch4_window7_224',    # [96 dim] Fastest, Good for debugging
    'small': 'swin_small_patch4_window7_224',   # [96 dim] Deeper than Tiny
    'base':  'swin_base_patch4_window7_224',    # [128 dim] Stronger performance
    'base384': 'swin_base_patch4_window12_384', # [128 dim] High-res pretrained (Heavier)
    'large': 'swin_large_patch4_window12_384'   # [192 dim] SOTA level (Requires 24GB+ VRAM)
}

# 👉 CHANGE THIS to switch models: 'tiny', 'small', 'base', 'base384', 'large'
MODEL_SELECT = 'base' 

# Safety check and assignment
if MODEL_SELECT not in BACKBONE_OPTIONS:
    raise ValueError(f"Invalid model selection: {MODEL_SELECT}. Choose from {list(BACKBONE_OPTIONS.keys())}")

BACKBONE_NAME = BACKBONE_OPTIONS[MODEL_SELECT]
print(f"🔹 Selected Backbone: {MODEL_SELECT.upper()} ({BACKBONE_NAME})")

# --- 3. Hyperparameters ---
IMG_SIZE = 1024         # [Strategy] Locator uses 1024x1024 input
BATCH_SIZE = 2         # [Advice] Base/Large: try 2.(for 16GB VRAM)
NUM_EPOCHS = 150        # Locator converges relatively fast
LEARNING_RATE = 1e-4    # Standard for Transformers
WEIGHT_DECAY = 1e-4     # AdamW standard

# --- 4. Strategy Settings ---
# [Twin-Swin Strategy]
USE_TWIN_ALIGNMENT = True  # Enable Feature Alignment (Needs MaskEncoder)
DILATE_MASK = True         # [Strategy] Dilate GT to prevent thin lines breaking at 1024

# --- 5. Loss Weights (Optimized for Locator) ---
# Weights for utils.loss.MattingLoss
LOSS_WEIGHTS = {
    'weight_bce': 1.0,     # Primary loss for Dilated GT
    'weight_iou': 1.0,     # Ensure coverage (Recall)
    'weight_ssim': 0.5,    # Structure consistency
    'weight_l1': 0.2,      # Pixel regression
    'weight_focal': 0.0,   # Disabled (BCE is sufficient for Dilated GT)
    'weight_grad': 0.0,    # Disabled (Avoid fitting artificial edges)
    'weight_feat': 0.2     # Feature Alignment weight
}

# --- 6. Paths ---
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
# DATASET_ROOT = "Datasets/DIS5K_Flat"  # Relative path
DATASET_ROOT = "/home/tec/Desktop/Project/Datasets/DIS5K_Flat" # Absolute path (Safer)

TRAIN_ROOT = os.path.join(DATASET_ROOT, 'train') # expects 'im' and 'gt' inside
VAL_ROOT = os.path.join(DATASET_ROOT, 'val')     # expects 'im' and 'gt' inside

# --- 7. Logging & Saving ---
# Auto-generate experiment name based on model selection
EXPERIMENT_NAME = f"TwinSwin_{MODEL_SELECT.capitalize()}_Loc1024"

CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'checkpoints', EXPERIMENT_NAME)
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs', EXPERIMENT_NAME)

# Ensure directories exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

BEST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'best_model.pth')
LAST_MODEL_PATH = os.path.join(CHECKPOINT_DIR, 'last_model.pth')