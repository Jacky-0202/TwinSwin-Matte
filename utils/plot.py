# utils/plot.py

import matplotlib.pyplot as plt
import torch
import numpy as np
import os

def to_cpu_numpy(data):
    if isinstance(data, torch.Tensor):
        return data.detach().cpu().numpy()
    if isinstance(data, list):
        if len(data) > 0 and isinstance(data[0], torch.Tensor):
            return np.array([x.detach().cpu().numpy() for x in data])
        return np.array(data)
    return np.array(data)

def plot_history(train_losses, val_losses, train_mse, val_mse, train_acc, val_acc, save_dir):
    """
    Plots: Loss, MSE, Accuracy.
    """
    train_losses = to_cpu_numpy(train_losses)
    val_losses = to_cpu_numpy(val_losses)
    train_mse = to_cpu_numpy(train_mse)
    val_mse = to_cpu_numpy(val_mse)
    train_acc = to_cpu_numpy(train_acc)
    val_acc = to_cpu_numpy(val_acc)
    
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(18, 5))

    # 1. Loss
    plt.subplot(1, 3, 1)
    if len(train_losses) > 0:
        plt.plot(epochs, train_losses, 'b-', label='Train Loss')
        plt.plot(epochs, val_losses, 'r-', label='Val Loss')
        plt.title('Loss Curve')
        plt.legend(); plt.grid(True)

    # 2. MSE (Lower is Better)
    plt.subplot(1, 3, 2)
    if len(train_mse) > 0:
        plt.plot(epochs, train_mse, 'b-', label='Train MSE')
        plt.plot(epochs, val_mse, 'm-', label='Val MSE')
        plt.title('MSE (Lower is Better)')
        plt.legend(); plt.grid(True)

    # 3. Accuracy (Higher is Better)
    plt.subplot(1, 3, 3)
    if len(train_acc) > 0:
        plt.plot(epochs, train_acc, 'b-', label='Train Acc')
        plt.plot(epochs, val_acc, 'g-', label='Val Acc')
        plt.title('Pixel Accuracy % (Higher is Better)')
        plt.ylabel('Accuracy (%)')
        plt.legend(); plt.grid(True)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, 'training_curves.png')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"📊 Training curves saved at: {save_path}")