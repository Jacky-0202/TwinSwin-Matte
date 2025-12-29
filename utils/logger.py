# utils/logger.py

import csv
import os
import torch

class CSVLogger:
    def __init__(self, save_dir, filename='training_log.csv'):
        """
        Args:
            save_dir (str): Directory where the log file will be saved.
            filename (str): Name of the CSV file.
        """
        self.save_dir = save_dir
        self.filepath = os.path.join(save_dir, filename)
        
        # [UPDATED] Define headers for TwinSwin-Matte task
        # Replaced Dice/mIoU with SAD, MSE, and MAD
        self.headers = [
            'Epoch', 'LR', 
            'Train_Loss', 'Train_MSE', 'Train_Acc', 
            'Val_Loss',   'Val_MSE',   'Val_Acc'
        ]
        
        # Ensure the directory exists
        os.makedirs(save_dir, exist_ok=True)
        
        # Create file and write headers (overwrite mode 'w')
        # Note: If you restart training, this will overwrite the old log!
        with open(self.filepath, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(self.headers)
        
        print(f"📝 Log file created at: {self.filepath}")

    def log(self, data):
        """
        Write a single row of data to the CSV.
        Automatically converts Tensors and formats floats to specific decimal places.
        
        Args:
            data (list): List of data values to write. 
                         Expected length: 10 (matching headers)
        """
        clean_data = []
        
        for x in data:
            # 1. Handle PyTorch Tensors (convert to float/int)
            if isinstance(x, torch.Tensor):
                x = x.detach().cpu().item()
            
            # 2. Handle Floats (Format precision)
            if isinstance(x, float):
                # Heuristic: If number is very small (like LR < 0.0001 or MSE), use scientific notation
                if 0 < abs(x) < 0.0001:
                    clean_data.append(f"{x:.4e}")
                # For SAD (can be large, e.g., 50.12), standard float is fine
                else:
                    clean_data.append(f"{x:.4f}")
            else:
                # Keep integers or strings as is
                clean_data.append(x)

        # Write the cleaned and formatted data to CSV
        with open(self.filepath, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(clean_data)