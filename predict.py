# predict.py

import os
import sys
import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import torch.nn.functional as F

# --- Project Setup ---
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

import config as config
from models.twin_swin_matte import TwinSwinMatteNet 

# --- 1. Settings ---
INPUT_DIR = 'test_data'         # Directory for input test images
OUTPUT_DIR = 'test_results'     # Directory for output results

# Model Path (Automatically point to the best model)
MODEL_PATH = config.BEST_MODEL_PATH

# Inference Size (Height, Width)
# We use the IMG_SIZE defined in config (e.g., 1024 or 768)
IMG_SIZE = config.IMG_SIZE

# --- 2. Visualization Helper ---
def generate_checkerboard(h, w, tile_size=20):
    """
    Generates a checkerboard background for visualizing alpha transparency.
    """
    checkerboard = np.zeros((h, w, 3), dtype=np.uint8)
    color1 = (255, 255, 255) # White
    color2 = (200, 200, 200) # Light Gray
    
    for y in range(0, h, tile_size):
        for x in range(0, w, tile_size):
            if (x // tile_size + y // tile_size) % 2 == 0:
                checkerboard[y:y+tile_size, x:x+tile_size] = color1
            else:
                checkerboard[y:y+tile_size, x:x+tile_size] = color2
    return checkerboard

# --- 3. Inference Pipeline ---
def process_image(img_path, model, device, transform):
    """
    Pipeline: 
    1. Load Image (RGB)
    2. Resize to IMG_SIZE
    3. Predict Alpha
    4. Upsample Alpha back to Original Size
    """
    # 1. Load original image
    # Image.MAX_IMAGE_PIXELS = None # Enable if dealing with massive images
    original_img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = original_img.size 
    
    # 2. Preprocessing (ToTensor + Normalize)
    # Unsqueeze to add batch dim: (1, 3, H, W)
    input_tensor = transform(original_img).unsqueeze(0).to(device)
    
    # 3. Resize input to Inference Size
    input_resized = F.interpolate(input_tensor, size=IMG_SIZE, mode='bilinear', align_corners=True)
    
    # 4. Model Inference
    with torch.no_grad():
        # Forward pass without GT (GT Encoder is idle)
        # Returns: pred_alpha, (stu_feats), (gt_feats=None)
        pred_alpha, _, _ = model(input_resized, gt_mask=None)
    
    # 5. Upsample Alpha back to Original Size
    # pred_alpha is (1, 1, H, W) with values 0.0 ~ 1.0
    output_alpha = F.interpolate(pred_alpha, size=(orig_h, orig_w), mode='bilinear', align_corners=True)
    
    # 6. Post-processing
    # Remove batch and channel dim -> (H, W)
    alpha_np = output_alpha.squeeze().cpu().numpy()
    
    # Ensure range 0~1
    alpha_np = np.clip(alpha_np, 0, 1)
        
    return alpha_np, original_img

def save_matting_results(alpha_map, original_img_pil, save_base_path):
    """
    Saves two files:
    1. _alpha.png: The grayscale alpha matte.
    2. _composite.png: The foreground composited on a checkerboard.
    """
    orig_np = np.array(original_img_pil) # RGB
    h, w, _ = orig_np.shape
    
    # Convert alpha to 0-255 uint8 for saving
    alpha_uint8 = (alpha_map * 255).astype(np.uint8)
    
    # --- 1. Save Alpha Matte ---
    cv2.imwrite(save_base_path + "_alpha.png", alpha_uint8)
    
    # --- 2. Save Composite (Checkerboard) ---
    # Create checkerboard background
    bg_checker = generate_checkerboard(h, w, tile_size=32)
    
    # Alpha blending: F * alpha + B * (1 - alpha)
    # Expand alpha to 3 channels: (H, W) -> (H, W, 3)
    alpha_3c = np.dstack((alpha_map, alpha_map, alpha_map))
    
    composite = (orig_np * alpha_3c + bg_checker * (1 - alpha_3c)).astype(np.uint8)
    # Convert RGB to BGR for OpenCV
    # cv2.imwrite(save_base_path + "_composite.png", cv2.cvtColor(composite, cv2.COLOR_RGB2BGR))

def main():
    # Setup
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    if not os.path.exists(INPUT_DIR):
        os.makedirs(INPUT_DIR, exist_ok=True)
        print(f"📁 Created '{INPUT_DIR}'. Please put test images here.")
        return

    device = config.DEVICE
    print(f"🚀 Loading Model from: {MODEL_PATH}")
    print(f"   Backbone: {config.BACKBONE}")
    print(f"   Inference Size: {IMG_SIZE}")

    # Initialize Model
    model = TwinSwinMatteNet(
        n_classes=config.NUM_CLASSES, 
        img_size=config.IMG_SIZE, 
        backbone_name=config.BACKBONE, 
        pretrained=False
    ).to(device)
    
    # Load Weights
    if os.path.exists(MODEL_PATH):
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device)
            state_dict = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
            model.load_state_dict(state_dict)
            model.eval()
            print("✅ Model weights loaded successfully.")
        except Exception as e:
            print(f"❌ Error loading weights: {e}")
            return
    else:
        print(f"❌ Weight file not found at: {MODEL_PATH}")
        print("   Please check config.py or train the model first.")
        return

    # Transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Get images
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(valid_ext)]
    
    if not image_files:
        print(f"⚠️ No images found in '{INPUT_DIR}'")
        return

    print(f"📂 Found {len(image_files)} images. Processing...")
    
    for img_name in tqdm(image_files):
        img_path = os.path.join(INPUT_DIR, img_name)
        save_base_name = os.path.splitext(img_name)[0]
        save_full_path = os.path.join(OUTPUT_DIR, save_base_name)
        
        try:
            alpha_map, orig_pil = process_image(img_path, model, device, transform)
            save_matting_results(alpha_map, orig_pil, save_full_path)
        except Exception as e:
            print(f"⚠️ Error processing {img_name}: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n✅ All Done! Results saved to '{OUTPUT_DIR}'")

if __name__ == "__main__":
    main()