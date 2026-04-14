#!/usr/bin/env python3
"""Convert DINOv3 safetensors weights to PyTorch .pth format."""
from safetensors.torch import load_file
import torch

SRC = "/home/cvpr_ug_5/.cache/torch/hub/checkpoints/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
DST = "/home/cvpr_ug_5/umesh/unsupervised_panoptic_segmentation/weights/dinov3_vitb16_official.pth"

sd = load_file(SRC)
torch.save(sd, DST)
print(f"Converted {len(sd)} keys -> {DST}")
