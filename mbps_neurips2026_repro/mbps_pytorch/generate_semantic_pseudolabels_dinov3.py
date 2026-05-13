#!/usr/bin/env python3
"""Generate semantic pseudo-labels using DINOv3 linear probing.

Follows the official DINOv3 evaluation protocol (refs/dinov3/) to train a
lightweight linear head (BN + 1x1 Conv) on frozen DINOv3 ViT-L/16 features
using Cityscapes GT labels. This replaces the K-means clustering approach in
generate_semantic_pseudolabels.py with a dramatically higher-quality method
(~70-75% mIoU vs ~25-35% for K-means).

Pipeline:
  1. Load DINOv3 ViT-L/16 backbone (frozen) from HuggingFace
  2. Extract multi-layer features via output_hidden_states
  3. Train a LinearHead on Cityscapes GT (following official protocol)
  4. Generate pseudo-labels via sliding window inference
  5. Evaluate on val set (mIoU)

Usage:
    # Full pipeline: train + evaluate + generate pseudo-labels
    python mbps_pytorch/generate_semantic_pseudolabels_dinov3.py \
        --mode all \
        --image_dir /data/cityscapes/leftImg8bit \
        --label_dir /data/cityscapes/gtFine \
        --output_dir /data/cityscapes/pseudo_semantic_dinov3

    # Generate only (with pre-trained head)
    python mbps_pytorch/generate_semantic_pseudolabels_dinov3.py \
        --mode generate \
        --image_dir /data/cityscapes/leftImg8bit \
        --label_dir /data/cityscapes/gtFine \
        --output_dir /data/cityscapes/pseudo_semantic_dinov3 \
        --head_checkpoint /data/cityscapes/pseudo_semantic_dinov3/best_head.pth

    # Evaluate only
    python mbps_pytorch/generate_semantic_pseudolabels_dinov3.py \
        --mode evaluate \
        --image_dir /data/cityscapes/leftImg8bit \
        --label_dir /data/cityscapes/gtFine \
        --output_dir /data/cityscapes/pseudo_semantic_dinov3 \
        --head_checkpoint /data/cityscapes/pseudo_semantic_dinov3/best_head.pth

Architecture reference: refs/dinov3/dinov3/eval/segmentation/
"""

import argparse
import json
import logging
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IGNORE_LABEL = 255
NUM_CLASSES_CITYSCAPES = 19

def _get_autocast_ctx(device: str):
    """Return an appropriate autocast context for the device."""
    if "cuda" in device:
        return torch.autocast("cuda", dtype=torch.bfloat16)
    elif "mps" in device:
        return torch.autocast("mps", dtype=torch.float16)
    else:
        import contextlib
        return contextlib.nullcontext()


CS_TRAINID_TO_NAME = {
    0: "road", 1: "sidewalk", 2: "building", 3: "wall", 4: "fence",
    5: "pole", 6: "traffic light", 7: "traffic sign", 8: "vegetation",
    9: "terrain", 10: "sky", 11: "person", 12: "rider", 13: "car",
    14: "truck", 15: "bus", 16: "train", 17: "motorcycle", 18: "bicycle",
}

# Layer indices for FOUR_EVEN_INTERVALS mode (0-indexed block numbers).
# Official DINOv3: ViT-L (24 blocks) -> [4, 11, 17, 23]
# ViT-B (12 blocks) -> [2, 5, 8, 11]
LAYER_INDICES = {
    12: [2, 5, 8, 11],       # ViT-B/S
    24: [4, 11, 17, 23],     # ViT-L
    40: [9, 19, 29, 39],     # ViT-g
}


# --------------------------------------------------------------------------- #
# LinearHead — ported from refs/dinov3/.../linear_head.py
# --------------------------------------------------------------------------- #
class LinearHead(nn.Module):
    """Linear segmentation head: BN + 1x1 Conv.

    Ported from refs/dinov3/dinov3/eval/segmentation/models/heads/linear_head.py.
    Uses BatchNorm2d instead of SyncBatchNorm (single-GPU training).
    """

    def __init__(
        self,
        in_channels: list,
        n_output_channels: int,
        use_batchnorm: bool = True,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.channels = sum(in_channels)
        self.n_output_channels = n_output_channels
        self.batchnorm_layer = (
            nn.BatchNorm2d(self.channels)
            if use_batchnorm
            else nn.Identity()
        )
        self.conv = nn.Conv2d(
            self.channels, n_output_channels, kernel_size=1, padding=0, stride=1,
        )
        self.dropout = nn.Dropout2d(dropout)
        # Official init: Normal(0, 0.01) for weight, 0 for bias
        nn.init.normal_(self.conv.weight, mean=0, std=0.01)
        nn.init.constant_(self.conv.bias, 0)

    def _transform_inputs(self, inputs: list) -> torch.Tensor:
        """Bilinear-interpolate all feature maps to match first map's size, then concat."""
        inputs = [
            F.interpolate(
                input=x,
                size=inputs[0].shape[2:],
                mode="bilinear",
                align_corners=False,
            )
            for x in inputs
        ]
        return torch.cat(inputs, dim=1)

    def forward(self, inputs: list) -> torch.Tensor:
        x = self._transform_inputs(inputs)
        x = self.dropout(x)
        x = self.batchnorm_layer(x)
        x = self.conv(x)
        return x

    def predict(self, inputs: list, rescale_to: tuple) -> torch.Tensor:
        """Predict without dropout, bilinear upsample to rescale_to."""
        x = self._transform_inputs(inputs)
        x = self.batchnorm_layer(x)
        x = self.conv(x)
        x = F.interpolate(input=x, size=rescale_to, mode="bilinear", align_corners=False)
        return x


# --------------------------------------------------------------------------- #
# DINOv3 Feature Extractor (HuggingFace)
# --------------------------------------------------------------------------- #
class DINOv3FeatureExtractor(nn.Module):
    """Extracts multi-layer spatial features from a frozen DINOv3 backbone.

    Mirrors refs/dinov3/dinov3/eval/utils.py:ModelWithIntermediateLayers,
    adapted for HuggingFace AutoModel output format.

    Critical: HF hidden_states are un-normed raw block outputs. The official
    DINOv3 get_intermediate_layers() applies self.norm to ALL extracted layers
    (see refs/dinov3/dinov3/models/vision_transformer.py:296-304). We replicate
    this by manually applying the model's LayerNorm.
    """

    def __init__(self, model_name: str, layer_indices: list, device: str = "cuda"):
        super().__init__()
        logger.info(f"Loading DINOv3 backbone: {model_name}")
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
        self.model.requires_grad_(False)

        self.layer_indices = layer_indices
        self.patch_size = self.model.config.patch_size
        self.embed_dim = self.model.config.hidden_size
        self.n_register = getattr(self.model.config, "num_register_tokens", 4)
        self.skip_tokens = 1 + self.n_register  # CLS + registers

        # Access the final LayerNorm for normalizing intermediate layers.
        # HuggingFace stores it differently depending on model class.
        # For Dinov2ForImageClassification: model.dinov2.layernorm
        # For Dinov2Model: model.layernorm
        if hasattr(self.model, "layernorm"):
            self.output_norm = self.model.layernorm
        elif hasattr(self.model, "dinov2") and hasattr(self.model.dinov2, "layernorm"):
            self.output_norm = self.model.dinov2.layernorm
        elif hasattr(self.model, "dinov3") and hasattr(self.model.dinov3, "layernorm"):
            self.output_norm = self.model.dinov3.layernorm
        else:
            logger.warning("Could not find output LayerNorm — features may be un-normed")
            self.output_norm = nn.Identity()

        n_blocks = self.model.config.num_hidden_layers
        logger.info(
            f"  embed_dim={self.embed_dim}, patch_size={self.patch_size}, "
            f"blocks={n_blocks}, registers={self.n_register}, "
            f"layer_indices={self.layer_indices}"
        )

    def forward(self, pixel_values: torch.Tensor) -> list:
        """Extract spatial features from specified layers.

        Args:
            pixel_values: (B, 3, H, W) normalized image tensor.

        Returns:
            List of (B, D, H_patches, W_patches) feature maps, one per layer.
        """
        B, _, H, W = pixel_values.shape
        h_patches = H // self.patch_size
        w_patches = W // self.patch_size

        outputs = self.model(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states
        # hidden_states[0] = embedding layer output
        # hidden_states[i+1] = block i output (0-indexed)

        features = []
        for layer_idx in self.layer_indices:
            # +1 because hidden_states[0] is the embedding layer
            hs = hidden_states[layer_idx + 1]  # (B, N_tokens, D)

            # Apply LayerNorm (matching official get_intermediate_layers norm=True)
            hs = self.output_norm(hs)

            # Strip CLS + register tokens → patch tokens only
            patch_tokens = hs[:, self.skip_tokens:, :]  # (B, N_patches, D)

            # Reshape to spatial format: (B, D, H_p, W_p)
            spatial = (
                patch_tokens
                .reshape(B, h_patches, w_patches, -1)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            features.append(spatial)

        return features


# --------------------------------------------------------------------------- #
# Segmentation Model (backbone + head)
# --------------------------------------------------------------------------- #
class SegmentationModel(nn.Module):
    def __init__(self, backbone: DINOv3FeatureExtractor, head: LinearHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        with torch.inference_mode():
            features = self.backbone(pixel_values)
        # Detach features so gradients only flow through head
        features = [f.detach() for f in features]
        return self.head(features)

    @torch.inference_mode()
    def predict(self, pixel_values: torch.Tensor, rescale_to: tuple) -> torch.Tensor:
        features = self.backbone(pixel_values)
        return self.head.predict(features, rescale_to=rescale_to)


# --------------------------------------------------------------------------- #
# Cityscapes Segmentation Dataset
# --------------------------------------------------------------------------- #
class CityscapesSegDataset(Dataset):
    """Cityscapes dataset for semantic segmentation with augmentations.

    Training augmentation follows official DINOv3 config:
      - Random resize: ratio in [0.5, 2.0]
      - Random crop: crop_size × crop_size
      - Random horizontal flip: p=0.5
      - ImageNet normalization
    """

    def __init__(
        self,
        image_dir: str,
        label_dir: str,
        is_train: bool = True,
        image_size: int = 512,
        crop_size: int = 512,
        scale_range: tuple = (0.5, 2.0),
        flip_prob: float = 0.5,
    ):
        self.is_train = is_train
        self.image_size = image_size
        self.crop_size = crop_size
        self.scale_range = scale_range
        self.flip_prob = flip_prob

        image_dir = Path(image_dir)
        label_dir = Path(label_dir)
        self.image_paths = sorted(image_dir.rglob("*_leftImg8bit.png"))
        if not self.image_paths:
            self.image_paths = sorted(image_dir.rglob("*.png"))

        # Match labels
        self.label_paths = []
        for img_path in self.image_paths:
            rel = img_path.relative_to(image_dir)
            base = str(rel).replace("_leftImg8bit.png", "")
            label_path = None
            for suffix in ["_gtFine_labelTrainIds.png", "_gtFine_labelIds.png"]:
                candidate = label_dir / (base + suffix)
                if candidate.exists():
                    label_path = candidate
                    break
            self.label_paths.append(label_path)

        # Filter out missing labels
        valid = [(i, l) for i, l in zip(self.image_paths, self.label_paths) if l is not None]
        if len(valid) < len(self.image_paths):
            logger.warning(
                f"Missing labels for {len(self.image_paths) - len(valid)}/{len(self.image_paths)} images"
            )
        self.image_paths, self.label_paths = zip(*valid) if valid else ([], [])
        logger.info(f"{'Train' if is_train else 'Val'} dataset: {len(self)} images")

        # Remap table for raw Cityscapes labelIds → trainIds
        self._CS_ID_TO_TRAIN = {
            7: 0, 8: 1, 11: 2, 12: 3, 13: 4, 17: 5, 19: 6, 20: 7,
            21: 8, 22: 9, 23: 10, 24: 11, 25: 12, 26: 13, 27: 14,
            28: 15, 31: 16, 32: 17, 33: 18,
        }

        # Normalization
        self.mean = torch.tensor(IMAGENET_MEAN).view(3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD).view(3, 1, 1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        label = np.array(Image.open(self.label_paths[idx]))

        # Remap raw labelIds to trainIds if needed
        if "labelTrainIds" not in str(self.label_paths[idx]):
            remapped = np.full_like(label, IGNORE_LABEL)
            for raw_id, train_id in self._CS_ID_TO_TRAIN.items():
                remapped[label == raw_id] = train_id
            label = remapped

        if self.is_train:
            img, label = self._train_augment(img, label)
        else:
            img, label = self._val_transform(img, label)

        return img, label

    def _train_augment(self, img: Image.Image, label: np.ndarray):
        """Random resize → random crop → random flip → normalize."""
        w, h = img.size

        # Random scale
        ratio = random.uniform(*self.scale_range)
        new_h, new_w = int(h * ratio), int(w * ratio)
        img = img.resize((new_w, new_h), Image.BILINEAR)
        label = np.array(
            Image.fromarray(label).resize((new_w, new_h), Image.NEAREST)
        )

        # Random crop to crop_size × crop_size
        crop_h, crop_w = self.crop_size, self.crop_size
        # Pad if needed
        pad_h = max(crop_h - new_h, 0)
        pad_w = max(crop_w - new_w, 0)
        if pad_h > 0 or pad_w > 0:
            img = np.array(img)
            img = np.pad(
                img,
                ((0, pad_h), (0, pad_w), (0, 0)),
                mode="constant",
                constant_values=0,
            )
            img = Image.fromarray(img)
            label = np.pad(
                label,
                ((0, pad_h), (0, pad_w)),
                mode="constant",
                constant_values=IGNORE_LABEL,
            )
            new_h += pad_h
            new_w += pad_w

        top = random.randint(0, new_h - crop_h)
        left = random.randint(0, new_w - crop_w)
        img = img.crop((left, top, left + crop_w, top + crop_h))
        label = label[top:top + crop_h, left:left + crop_w]

        # Random horizontal flip
        if random.random() < self.flip_prob:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = np.fliplr(label).copy()

        # To tensor + normalize
        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        label_tensor = torch.from_numpy(label.copy()).long()

        return img_tensor, label_tensor

    def _val_transform(self, img: Image.Image, label: np.ndarray):
        """Resize short side to image_size, normalize. Keep original aspect."""
        w, h = img.size
        if h < w:
            new_h = self.image_size
            new_w = int(self.image_size * w / h + 0.5)
        else:
            new_w = self.image_size
            new_h = int(self.image_size * h / w + 0.5)

        # Make dimensions multiples of patch_size (16)
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16

        img = img.resize((new_w, new_h), Image.BILINEAR)

        img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float() / 255.0
        img_tensor = (img_tensor - self.mean) / self.std
        label_tensor = torch.from_numpy(label.copy()).long()

        return img_tensor, label_tensor


# --------------------------------------------------------------------------- #
# Sliding Window Inference
# Ported from refs/dinov3/dinov3/eval/segmentation/inference.py
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def slide_inference(
    model: SegmentationModel,
    image: torch.Tensor,
    num_classes: int,
    crop_size: tuple = (512, 512),
    stride: tuple = (341, 341),
) -> torch.Tensor:
    """Sliding window inference with overlap averaging.

    Args:
        model: SegmentationModel in eval mode.
        image: (1, 3, H, W) normalized tensor on device.
        num_classes: Number of output classes.
        crop_size: (h_crop, w_crop) for sliding windows.
        stride: (h_stride, w_stride) for sliding windows.

    Returns:
        (1, num_classes, H, W) averaged logits on CPU.
    """
    h_stride, w_stride = stride
    h_crop, w_crop = crop_size
    _, _, h_img, w_img = image.shape

    h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
    w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

    preds = image.new_zeros((1, num_classes, h_img, w_img)).cpu()
    count_mat = image.new_zeros((1, 1, h_img, w_img)).to(torch.int16).cpu()

    for h_idx in range(h_grids):
        for w_idx in range(w_grids):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            crop_img = image[:, :, y1:y2, x1:x2]
            crop_pred = model.predict(crop_img, rescale_to=crop_img.shape[2:])

            preds[:, :, y1:y2, x1:x2] += crop_pred.cpu()
            count_mat[:, :, y1:y2, x1:x2] += 1

    assert (count_mat == 0).sum() == 0, "Some pixels were not covered by any window"
    return preds / count_mat


# --------------------------------------------------------------------------- #
# Learning Rate Scheduler
# --------------------------------------------------------------------------- #
def build_scheduler(optimizer, warmup_iters: int, total_iters: int):
    """Warmup + cosine decay scheduler matching official DINOv3 protocol."""
    return torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=optimizer.param_groups[0]["lr"],
        total_steps=total_iters,
        pct_start=warmup_iters / total_iters,
        anneal_strategy="cos",
        div_factor=1e6,   # start lr ≈ 0
        final_div_factor=1e6,  # end lr ≈ 0
    )


# --------------------------------------------------------------------------- #
# Checkpoint utilities
# --------------------------------------------------------------------------- #
def save_checkpoint(head: LinearHead, optimizer, step: int, miou: float, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "step": step,
            "miou": miou,
        },
        path,
    )
    logger.info(f"Saved checkpoint: {path} (step={step}, mIoU={miou:.4f})")


def load_checkpoint(head: LinearHead, path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    head.load_state_dict(ckpt["head_state_dict"])
    logger.info(f"Loaded head from {path} (step={ckpt['step']}, mIoU={ckpt['miou']:.4f})")
    return ckpt


# --------------------------------------------------------------------------- #
# Helper: resolve layer indices
# --------------------------------------------------------------------------- #
def get_layer_indices(layer_mode: str, n_blocks: int) -> list:
    if layer_mode == "last":
        return [n_blocks - 1]
    elif layer_mode == "four_even":
        if n_blocks in LAYER_INDICES:
            return LAYER_INDICES[n_blocks]
        return [i * (n_blocks // 4) - 1 for i in range(1, 5)]
    elif layer_mode == "four_last":
        return list(range(n_blocks - 4, n_blocks))
    else:
        raise ValueError(f"Unknown layer_mode: {layer_mode}")


# --------------------------------------------------------------------------- #
# Build model
# --------------------------------------------------------------------------- #
def build_model(
    model_name: str,
    layer_mode: str,
    num_classes: int,
    device: str,
) -> SegmentationModel:
    """Build frozen backbone + trainable linear head."""
    # Determine layer indices from model config
    tmp_model = AutoModel.from_pretrained(model_name)
    n_blocks = tmp_model.config.num_hidden_layers
    embed_dim = tmp_model.config.hidden_size
    del tmp_model

    layer_indices = get_layer_indices(layer_mode, n_blocks)
    in_channels = [embed_dim] * len(layer_indices)

    backbone = DINOv3FeatureExtractor(model_name, layer_indices, device)
    head = LinearHead(
        in_channels=in_channels,
        n_output_channels=num_classes,
        use_batchnorm=True,
        dropout=0.1,
    )

    model = SegmentationModel(backbone, head).to(device)

    # Freeze backbone
    for p in model.backbone.parameters():
        p.requires_grad = False

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    logger.info(f"Model: {total:,} total params, {trainable:,} trainable (head only)")

    return model


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_linear_head(
    model_name: str,
    image_dir: str,
    label_dir: str,
    output_dir: str,
    num_classes: int = 19,
    layer_mode: str = "last",
    batch_size: int = 4,
    total_iters: int = 20000,
    warmup_iters: int = 1500,
    lr: float = 1e-3,
    weight_decay: float = 1e-3,
    eval_interval: int = 2000,
    image_size: int = 512,
    crop_size: int = 512,
    device: str = "cuda",
) -> float:
    """Train a linear segmentation head on Cityscapes GT labels.

    Returns best mIoU achieved during training.
    """
    os.makedirs(output_dir, exist_ok=True)
    model = build_model(model_name, layer_mode, num_classes, device)

    # Datasets
    train_dataset = CityscapesSegDataset(
        image_dir=os.path.join(image_dir, "train"),
        label_dir=os.path.join(label_dir, "train"),
        is_train=True,
        image_size=image_size,
        crop_size=crop_size,
        scale_range=(0.5, 2.0),
        flip_prob=0.5,
    )
    val_dataset = CityscapesSegDataset(
        image_dir=os.path.join(image_dir, "val"),
        label_dir=os.path.join(label_dir, "val"),
        is_train=False,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Optimizer + scheduler (official: AdamW, lr=1e-3, wd=1e-3)
    optimizer = torch.optim.AdamW(
        model.head.parameters(),
        lr=lr,
        betas=(0.9, 0.999),
        weight_decay=weight_decay,
    )
    scheduler = build_scheduler(optimizer, warmup_iters, total_iters)
    criterion = nn.CrossEntropyLoss(ignore_index=IGNORE_LABEL)

    # Training loop (iteration-based)
    model.head.train()
    best_miou = 0.0
    global_step = 0
    train_iter = iter(train_loader)
    epoch = 0
    running_loss = 0.0

    logger.info(f"Starting training: {total_iters} iters, bs={batch_size}, lr={lr}")
    start_time = time.time()

    while global_step < total_iters:
        try:
            images, labels = next(train_iter)
        except StopIteration:
            epoch += 1
            train_iter = iter(train_loader)
            images, labels = next(train_iter)

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        with _get_autocast_ctx(str(images.device)):
            logits = model(images)  # (B, C, H_patch, W_patch)
            # Upsample logits to label resolution
            if logits.shape[-2:] != labels.shape[-2:]:
                logits = F.interpolate(
                    logits, size=labels.shape[-2:],
                    mode="bilinear", align_corners=False,
                )
            loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()
        scheduler.step()
        global_step += 1
        running_loss += loss.item()

        if global_step % 50 == 0:
            avg_loss = running_loss / 50
            elapsed = time.time() - start_time
            iters_per_sec = global_step / elapsed
            eta_sec = (total_iters - global_step) / max(iters_per_sec, 1e-6)
            logger.info(
                f"Step {global_step}/{total_iters}  "
                f"loss={avg_loss:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}  "
                f"epoch={epoch}  "
                f"ETA={eta_sec / 60:.1f}min"
            )
            running_loss = 0.0

        if global_step % eval_interval == 0:
            miou = evaluate_model(
                model, val_loader, num_classes, device,
                crop_size=(crop_size, crop_size),
                stride=(crop_size * 2 // 3, crop_size * 2 // 3),
            )
            if miou > best_miou:
                best_miou = miou
                save_checkpoint(
                    model.head, optimizer, global_step, miou,
                    os.path.join(output_dir, "best_head.pth"),
                )
            model.head.train()

    # Final evaluation
    miou = evaluate_model(
        model, val_loader, num_classes, device,
        crop_size=(crop_size, crop_size),
        stride=(crop_size * 2 // 3, crop_size * 2 // 3),
    )
    if miou > best_miou:
        best_miou = miou
    save_checkpoint(
        model.head, optimizer, global_step, miou,
        os.path.join(output_dir, "final_head.pth"),
    )

    elapsed = time.time() - start_time
    logger.info(f"Training complete in {elapsed / 60:.1f} min. Best mIoU: {best_miou:.4f}")

    # Save config
    config = {
        "model_name": model_name,
        "layer_mode": layer_mode,
        "num_classes": num_classes,
        "total_iters": total_iters,
        "best_miou": best_miou,
        "batch_size": batch_size,
        "lr": lr,
        "crop_size": crop_size,
    }
    with open(os.path.join(output_dir, "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

    return best_miou


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def evaluate_model(
    model: SegmentationModel,
    val_loader: DataLoader,
    num_classes: int,
    device: str,
    crop_size: tuple = (512, 512),
    stride: tuple = (341, 341),
) -> float:
    """Evaluate model with sliding window inference, return mIoU."""
    model.eval()

    total_intersect = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)

    for images, labels in tqdm(val_loader, desc="Evaluating", leave=False):
        images = images.to(device)
        gt_h, gt_w = labels.shape[-2:]

        # Sliding window inference
        with _get_autocast_ctx(str(images.device)):
            preds = slide_inference(model, images, num_classes, crop_size, stride)

        pred_labels = preds.softmax(dim=1).argmax(dim=1).squeeze(0)  # (H, W)

        # Resize prediction to GT resolution if needed
        if pred_labels.shape != (gt_h, gt_w):
            pred_labels = (
                F.interpolate(
                    pred_labels.float().unsqueeze(0).unsqueeze(0),
                    size=(gt_h, gt_w),
                    mode="nearest",
                )
                .squeeze(0)
                .squeeze(0)
                .long()
            )

        labels = labels.squeeze(0)  # (H, W)

        # Per-class intersection and union
        for c in range(num_classes):
            pred_c = pred_labels.cpu() == c
            gt_c = labels == c
            valid = labels != IGNORE_LABEL
            total_intersect[c] += (pred_c & gt_c & valid).sum().float()
            total_union[c] += ((pred_c | gt_c) & valid).sum().float()

    # Compute mIoU
    iou = total_intersect / (total_union + 1e-10)
    valid_classes = total_union > 0
    miou = iou[valid_classes].mean().item()

    logger.info("Per-class IoU:")
    for c in range(num_classes):
        name = CS_TRAINID_TO_NAME.get(c, "???")
        if total_union[c] > 0:
            logger.info(f"  {c:2d} {name:15s}: {iou[c].item():.4f}")
        else:
            logger.info(f"  {c:2d} {name:15s}: N/A (no pixels)")
    logger.info(f"  mIoU = {miou:.4f} ({miou * 100:.2f}%)")

    model.head.train()
    return miou


# --------------------------------------------------------------------------- #
# Pseudo-Label Generation
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def generate_pseudolabels(
    model: SegmentationModel,
    image_dir: str,
    output_dir: str,
    num_classes: int,
    split: str = "train",
    image_size: int = 512,
    crop_size: tuple = (512, 512),
    stride: tuple = (341, 341),
    device: str = "cuda",
):
    """Generate semantic pseudo-labels for all images via sliding window inference.

    Output: PNG files with uint8 values 0-18 (Cityscapes trainIDs).
    """
    model.eval()
    split_dir = Path(image_dir) / split
    out_dir = Path(output_dir) / split

    image_paths = sorted(split_dir.rglob("*_leftImg8bit.png"))
    if not image_paths:
        image_paths = sorted(split_dir.rglob("*.png"))
    logger.info(f"Generating pseudo-labels for {len(image_paths)} images ({split})")

    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1).to(device)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1).to(device)

    skipped = 0
    for img_path in tqdm(image_paths, desc=f"Generating ({split})"):
        rel_path = img_path.relative_to(split_dir)
        out_path = out_dir / rel_path
        if out_path.exists():
            skipped += 1
            continue

        img = Image.open(img_path).convert("RGB")
        orig_w, orig_h = img.size

        # Resize: short side to image_size, preserve aspect, multiple of 16
        if orig_h < orig_w:
            new_h = image_size
            new_w = int(image_size * orig_w / orig_h + 0.5)
        else:
            new_w = image_size
            new_h = int(image_size * orig_h / orig_w + 0.5)
        new_h = (new_h // 16) * 16
        new_w = (new_w // 16) * 16

        img_resized = img.resize((new_w, new_h), Image.BILINEAR)

        # To tensor + normalize
        img_tensor = (
            torch.from_numpy(np.array(img_resized))
            .permute(2, 0, 1)
            .float()
            .unsqueeze(0)
            / 255.0
        ).to(device)
        img_tensor = (img_tensor - mean) / std

        # Sliding window inference
        with _get_autocast_ctx(device):
            preds = slide_inference(model, img_tensor, num_classes, crop_size, stride)
        pred_labels = preds.softmax(dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
        pred_labels = pred_labels.astype(np.uint8)

        # Resize back to original resolution
        pred_img = Image.fromarray(pred_labels, mode="L")
        pred_img = pred_img.resize((orig_w, orig_h), Image.NEAREST)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        pred_img.save(str(out_path))

    if skipped > 0:
        logger.info(f"Skipped {skipped} existing files")
    logger.info(f"Pseudo-labels saved to {out_dir}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="DINOv3 linear probing for semantic pseudo-label generation"
    )
    # Mode
    parser.add_argument(
        "--mode", choices=["train", "evaluate", "generate", "all"],
        default="all",
        help="Pipeline stage to run (default: all)",
    )
    # Model
    parser.add_argument(
        "--model_name", type=str,
        default="facebook/dinov3-vitl16-pretrain-lvd1689m",
        help="HuggingFace model name for DINOv3 backbone",
    )
    parser.add_argument(
        "--layer_mode", choices=["last", "four_even", "four_last"],
        default="last",
        help="Which layers to extract: 'last' (official linear eval), "
             "'four_even' (layers at 25/50/75/100%% depth)",
    )
    # Paths
    parser.add_argument(
        "--image_dir", type=str, required=True,
        help="Cityscapes leftImg8bit/ root (contains train/, val/)",
    )
    parser.add_argument(
        "--label_dir", type=str, required=True,
        help="Cityscapes gtFine/ root (contains train/, val/)",
    )
    parser.add_argument(
        "--output_dir", type=str, required=True,
        help="Output directory for checkpoints and pseudo-labels",
    )
    parser.add_argument(
        "--head_checkpoint", type=str, default=None,
        help="Path to pre-trained head checkpoint (for generate/evaluate modes)",
    )
    # Training hyperparams
    parser.add_argument("--num_classes", type=int, default=19)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--total_iters", type=int, default=20000)
    parser.add_argument("--warmup_iters", type=int, default=1500)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--eval_interval", type=int, default=2000)
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--crop_size", type=int, default=512)
    # Inference
    parser.add_argument("--stride", type=int, default=None,
                        help="Sliding window stride (default: 2/3 of crop_size)")
    parser.add_argument("--generate_split", type=str, default="train",
                        help="Which split to generate pseudo-labels for")
    # Device
    parser.add_argument("--device", type=str, default="auto")

    args = parser.parse_args()

    # Resolve device
    if args.device == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    # Resolve stride
    stride_val = args.stride if args.stride else args.crop_size * 2 // 3
    crop = (args.crop_size, args.crop_size)
    stride = (stride_val, stride_val)

    # ---- TRAIN ---- #
    if args.mode in ("train", "all"):
        train_linear_head(
            model_name=args.model_name,
            image_dir=args.image_dir,
            label_dir=args.label_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            layer_mode=args.layer_mode,
            batch_size=args.batch_size,
            total_iters=args.total_iters,
            warmup_iters=args.warmup_iters,
            lr=args.lr,
            weight_decay=args.weight_decay,
            eval_interval=args.eval_interval,
            image_size=args.image_size,
            crop_size=args.crop_size,
            device=device,
        )

    # ---- EVALUATE ---- #
    if args.mode in ("evaluate", "all"):
        head_path = args.head_checkpoint or os.path.join(args.output_dir, "best_head.pth")
        if not os.path.exists(head_path):
            logger.error(f"Head checkpoint not found: {head_path}")
            return

        model = build_model(args.model_name, args.layer_mode, args.num_classes, device)
        load_checkpoint(model.head, head_path)
        model.eval()

        val_dataset = CityscapesSegDataset(
            image_dir=os.path.join(args.image_dir, "val"),
            label_dir=os.path.join(args.label_dir, "val"),
            is_train=False,
            image_size=args.image_size,
        )
        val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=4)
        evaluate_model(model, val_loader, args.num_classes, device, crop, stride)

    # ---- GENERATE ---- #
    if args.mode in ("generate", "all"):
        head_path = args.head_checkpoint or os.path.join(args.output_dir, "best_head.pth")
        if not os.path.exists(head_path):
            logger.error(f"Head checkpoint not found: {head_path}")
            return

        model = build_model(args.model_name, args.layer_mode, args.num_classes, device)
        load_checkpoint(model.head, head_path)
        model.eval()

        generate_pseudolabels(
            model=model,
            image_dir=args.image_dir,
            output_dir=args.output_dir,
            num_classes=args.num_classes,
            split=args.generate_split,
            image_size=args.image_size,
            crop_size=crop,
            stride=stride,
            device=device,
        )


if __name__ == "__main__":
    main()
