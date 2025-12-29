# utils/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

class MattingDataset(Dataset):
    def __init__(self, root_dir, mode='train', img_size=512):
        """
        Dataset for TwinSwin-Matte (DIS5K Structure).
        
        Args:
            root_dir (str): Path to dataset root (e.g., 'Datasets/DIS5K_Flat')
            mode (str): 'train', 'val', or 'test'
            img_size (int): Target size for resizing/cropping
        
        Structure Expected:
            root_dir/
              ├── train/
              │   ├── im/ (Images .jpg/.png)
              │   └── gt/ (Masks .png)
              └── val/ ...
        """
        self.root_dir = root_dir
        self.mode = mode
        self.img_size = img_size
        
        # 1. Construct Paths based on new structure
        self.img_folder = os.path.join(root_dir, mode, 'im')
        self.mask_folder = os.path.join(root_dir, mode, 'gt')
        
        # 2. Get File List
        if not os.path.exists(self.img_folder):
            raise FileNotFoundError(f"Image folder not found: {self.img_folder}")
            
        valid_ext = ('.jpg', '.jpeg', '.png')
        self.image_files = sorted([f for f in os.listdir(self.img_folder) if f.lower().endswith(valid_ext)])
        
        print(f"[{mode.upper()}] Loaded {len(self.image_files)} images from {self.img_folder}")

        # 3. Base Transforms (ImageNet Norm)
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # --- A. Load Image & Mask ---
        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        
        # Determine Mask Name (DIS5K masks are usually .png)
        # Handle extension mismatch (e.g. image.jpg -> mask.png)
        file_stem = os.path.splitext(img_name)[0]
        mask_name = file_stem + '.png'
        mask_path = os.path.join(self.mask_folder, mask_name)
        
        # Fallback if mask is jpg (rare but possible)
        if not os.path.exists(mask_path):
             mask_path = os.path.join(self.mask_folder, file_stem + '.jpg')

        # Load using PIL (RGB for Image, L for Mask)
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L') # L = 8-bit pixels, black and white

        # --- B. Augmentation & Transform ---
        
        if self.mode == 'train':
            # 1. Random Resize (0.5x to 1.5x)
            # Matting needs robust scale invariance
            scale = np.random.uniform(0.75, 1.25)
            # Enforce size constraints to avoid crash
            new_w = int(image.width * scale)
            new_h = int(image.height * scale)
            
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.BILINEAR) # Mask uses Bilinear to preserve gradients

            # 2. Random Crop
            # Pad if smaller than crop size
            if new_w < self.img_size or new_h < self.img_size:
                pad_w = max(0, self.img_size - new_w)
                pad_h = max(0, self.img_size - new_h)
                image = TF.pad(image, (0, 0, pad_w, pad_h), fill=0)
                mask = TF.pad(mask, (0, 0, pad_w, pad_h), fill=0) # Pad background with 0
            
            # Crop
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(self.img_size, self.img_size))
            image = TF.crop(image, i, j, h, w)
            mask = TF.crop(mask, i, j, h, w)
            
            # 3. Random Horizontal Flip
            if np.random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
                
            # 4. Color Jitter (Only for Image)
            # Helps model handle different lighting conditions
            if np.random.random() > 0.2:
                color_tf = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05)
                image = color_tf(image)

        else:
            # Validation Mode:
            # Resize long side to INFER_SIZE or keep original?
            # For simplicity in batch processing, let's resize to fixed size.
            # Ideally, TwinSwin works on arbitrary size, but DataLoader needs consistent batch shapes.
            # If batch_size=1, we can skip resizing, but here we resize to config.IMG_SIZE for safety.
            # BETTER: Resize to closest multiple of 32 (Swin requirement)
            
            w, h = image.size
            # Resize so short side is img_size
            if w < h:
                new_w = self.img_size
                new_h = int(h * (self.img_size / w))
            else:
                new_h = self.img_size
                new_w = int(w * (self.img_size / h))
            
            image = TF.resize(image, (new_h, new_w), interpolation=Image.BILINEAR)
            mask = TF.resize(mask, (new_h, new_w), interpolation=Image.BILINEAR)

        # --- C. ToTensor & Normalize ---
        image = TF.to_tensor(image) # 0.0 ~ 1.0
        image = self.normalize(image)
        
        mask = TF.to_tensor(mask)   # 0.0 ~ 1.0 (Crucial: DO NOT THRESHOLD!)
        
        return image, mask