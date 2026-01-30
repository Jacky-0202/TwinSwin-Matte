# TwinSwin-Matte

TwinSwin-Matte is a high-resolution image matting framework that synergizes the global modeling power of Swin Transformers with the local precision of CNNs. Designed for H200 GPU environments, it leverages a Teacher-Student distillation strategy and a Decoupled Resolution approach to achieve SOTA-level edge accuracy on the DIS5K dataset.

---
### 🚀 Key Innovations

- Decoupled Resolution Strategy: 
Operates the Swin backbone at a mathematically "Safe Size" ($896\times896$) to maintain window-attention integrity while delivering final outputs at $1024\times1024$.
- Twin Alignment (Distillation): 
A CNN-based MaskEncoder (Teacher) guides the TwinSwin (Student) through feature-level alignment, infusing local inductive bias into the Transformer architecture.
- Hybrid Architecture: 
Combines a hierarchical Swin Transformer encoder with a multi-stage CNN decoder for superior semantic understanding and boundary refinement.
- H200 Optimized: Fully integrated with AMP (Mixed Precision) and Gradient Accumulation to maximize throughput on NVIDIA H200 hardware.

---
### 🧠 Model Architecture

The framework consists of two primary components:

Student (TwinSwinUNet): Swin-Base Encoder + Refined CNN Decoder.
Teacher (MaskEncoder): A Pure FPN (Feature Pyramid Network) extracting structural priors from ground-truth masks.

![Model Architecture](figures/twin_swin_matte_002.png)

---
### 🛠️ Quick Start

1. Installation

```bash
git clone https://github.com/your-repo/TwinSwin-Matte.git
cd TwinSwin-Matte
pip install -r requirements.txt
```

2. Training
Configure your dataset paths in config.py and run:

```bash
# To start fresh
python train.py
```

3. Evaluation
Evaluate the model performance on the standard DIS-TE1 test set:

```bash
python eval.py
```

4. Inference
Run prediction on single images or a directory:

```bash
python predict.py
```

---
📜 References

- Swin Transformer: [Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030)
- DIS5K Dataset: [Highly Accurate Dichotomous Image Segmentation](https://arxiv.org/abs/2203.03041)
- Swin-Unet: [Swin-Unet: Unet-like Pure Transformer for Medical Image Segmentation](https://arxiv.org/abs/2105.05537)

