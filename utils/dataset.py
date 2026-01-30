# utils/dataset.py

import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import Config

class MattingDataset(Dataset):
    def __init__(self, root_dir, mode='train', target_size=1024, schema='standard'):
        """
        Universal Matting Dataset Loader.
        Adapts to different folder structures (DIS5K, COD10K, etc.) based on 'schema'.
        
        Args:
            root_dir (str): Full path to the specific dataset subset (e.g., .../DIS5K/DIS-TR).
            mode (str): 'train' (enables augmentation) or 'val' (deterministic).
            target_size (int): Resolution.
            schema (str): 'standard' (im/gt) or 'cod10k' (Image/GT) or 'hrsod' (imgs/masks).
        """
        self.root_dir = root_dir
        self.mode = mode
        self.target_size = target_size
        self.dilate_mask = Config.DILATE_MASK
        
        # =========================================================
        # 1. Define Folder Names based on Schema
        # =========================================================
        if schema == 'standard':
            # DIS5K, HRS10K, HRSOD (Standard)
            img_dir_name, gt_dir_name = 'im', 'gt'
        elif schema == 'cod10k':
            # COD10K (Often uses Image/GT or Imgs/GT)
            # We try to auto-detect common variations for COD10K
            if os.path.exists(os.path.join(root_dir, 'Image')):
                img_dir_name, gt_dir_name = 'Image', 'GT'
            elif os.path.exists(os.path.join(root_dir, 'Imgs')):
                img_dir_name, gt_dir_name = 'Imgs', 'GT'
            else:
                # Fallback to standard if capital folder not found
                img_dir_name, gt_dir_name = 'im', 'gt'
        elif schema == 'hrsod':
            img_dir_name, gt_dir_name = 'imgs', 'masks'
        else:
            raise ValueError(f"Unknown schema: {schema}")

        # =========================================================
        # 2. Construct & Validate Paths
        # =========================================================
        self.img_folder = os.path.join(root_dir, img_dir_name)
        self.mask_folder = os.path.join(root_dir, gt_dir_name)
        
        if not os.path.exists(self.img_folder) or not os.path.exists(self.mask_folder):
            raise FileNotFoundError(
                f"❌ Dataset structure error! \n"
                f"   Schema: '{schema}' \n"
                f"   Looking for folders inside: {root_dir}\n"
                f"   - Images: {img_dir_name} (Found: {os.path.exists(self.img_folder)})\n"
                f"   - Masks:  {gt_dir_name} (Found: {os.path.exists(self.mask_folder)})\n"
                f"   Please check your config.py TRAIN_SET/VAL_SET paths."
            )
            
        # 3. Load File List
        valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif')
        self.image_files = sorted([f for f in os.listdir(self.img_folder) if f.lower().endswith(valid_ext)])
        
        print(f"[{mode.upper()}] Loaded {len(self.image_files)} images from: {self.img_folder}")

        # 4. Define Transformations (Albumentations)
        if mode == 'train':
            self.transform = A.Compose([
                # Global Context & Geometry
                A.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
                A.HorizontalFlip(p=0.5),
                # Color
                A.RandomBrightnessContrast(p=0.2),
                # Resize & Pad
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size, 
                    min_width=target_size, 
                    position='center',
                    border_mode=0,
                ),
                # Normalize
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])
        else:
            # Val/Test
            self.transform = A.Compose([
                A.LongestMaxSize(max_size=target_size),
                A.PadIfNeeded(
                    min_height=target_size, 
                    min_width=target_size, 
                    position='center',
                    border_mode=0,
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        # 1. Read Image
        img_name = self.image_files[index]
        img_path = os.path.join(self.img_folder, img_name)
        
        image = cv2.imread(img_path)
        if image is None:
            return self.__getitem__((index + 1) % len(self)) 
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # 2. Read Mask (Smart Extension)
        file_stem = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_folder, file_stem + '.png')
        if not os.path.exists(mask_path):
            mask_path = os.path.join(self.mask_folder, file_stem + '.jpg')
            
        mask = cv2.imread(mask_path, 0)
        if mask is None:
            return self.__getitem__((index + 1) % len(self))

        # 3. Dynamic GT Dilation (Train Only)
        if self.dilate_mask and self.mode == 'train':
            kernel = np.ones((3, 3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=1)

        # 4. Augmentation
        augmented = self.transform(image=image, mask=mask)
        img_tensor = augmented['image']
        mask_tensor = augmented['mask'].float() / 255.0
        mask_tensor = mask_tensor.unsqueeze(0)
        
        return img_tensor, mask_tensor