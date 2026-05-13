#!/usr/bin/env python3
"""Train SINDER epsilon perturbations for DINOv2 ViT-B/14.

SINDER repairs "singular defects" in DINOv2 by learning small perturbations
to singular values of weight matrices. This script adapts the original SINDER
training (designed for ViT-g/14 + ImageNet) to work with:
  - ViT-B/14 (regular MLP, not SwiGLU)
  - Cityscapes images (no ImageNet needed - loss is self-supervised)
  - MPS (Apple Silicon) or CUDA

The training is very lightweight: only ~36K epsilon parameters are learned.

Usage:
    python mbps_pytorch/train_sinder_vitb14.py \
        --cityscapes_root /path/to/cityscapes \
        --output_dir refs/cause/checkpoint \
        --max_iter 10000 \
        --device mps
"""

import argparse
import os
import sys
from glob import glob
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

# Add SINDER to path
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
SINDER_DIR = str(PROJECT_ROOT / "refs" / "sinder")
sys.path.insert(0, SINDER_DIR)

os.environ["XFORMERS_DISABLED"] = "1"

import sinder
from sinder import (
    replace_back,
    replace_linear_addition_noqk,
)
from sinder.neighbor_loss import check_anomaly_theoretical


# --------------------------------------------------------------------------- #
# Adapted singular defect computation for regular MLP (fc1 + GELU + fc2)
# --------------------------------------------------------------------------- #
def fc1_act(blk, x):
    """Forward through fc1 + activation (GELU) — regular MLP equivalent of w12."""
    with torch.no_grad():
        # Ensure computation on CPU for lstsq compatibility
        fc1_w = blk.mlp.fc1.weight.detach().cpu()
        fc1_b = blk.mlp.fc1.bias.detach().cpu()
        x_cpu = x.cpu()
        out = F.linear(x_cpu, fc1_w, fc1_b)
        out = F.gelu(out)
        return out


def anomaly_dir_mlp_ls_regular(blk, identity=False, bias=False, centered=False,
                                homogeneous=False, bias_ls=False):
    """Compute MLP singular defect direction for regular MLP (fc1+GELU+fc2).

    Adapted from sinder.singular_defect.anomaly_dir_mlp_ls which assumes SwiGLU.
    All computation done on CPU since linalg_lstsq/svd not supported on MPS.
    """
    with torch.no_grad():
        N = blk.ls2.gamma.shape[0]
        M = blk.mlp.fc2.weight.shape[1]  # hidden_dim (fc2: hidden -> out)
        cpu = torch.device("cpu")

        A4 = torch.diag(blk.ls2.gamma.detach().cpu())
        A3 = blk.mlp.fc2.weight.detach().cpu()
        B3 = blk.mlp.fc2.bias.detach().cpu()

        # Approximate fc1+GELU nonlinearity via least squares (on CPU)
        X = torch.randn(100000, N, device=cpu)
        Y = fc1_act(blk, X)  # already returns CPU tensor
        if bias_ls:
            X_one = torch.cat((X, torch.ones(100000, 1)), dim=1)
        else:
            X_one = X
        sol = torch.linalg.lstsq(X_one, Y)
        if bias_ls:
            A2 = sol.solution.T[:, :-1]
            B2 = sol.solution.T[:, -1]
        else:
            A2 = sol.solution.T
            B2 = torch.zeros(M)

        A1 = torch.diag(blk.norm2.weight.detach().cpu())
        B1 = blk.norm2.bias.detach().cpu()
        A0 = (torch.eye(N) - 1 / N * torch.ones(N, N))
        A = A4 @ A3 @ A2 @ A1

        if centered:
            A = A @ A0
        B = A4 @ (A3 @ (A2 @ B1)) + A4 @ (A3 @ B2) + A4 @ B3

        if bias:
            A = torch.cat((A, B[:, None]), dim=1)
            if homogeneous:
                onehot = torch.cat(
                    (torch.zeros_like(B), torch.ones(1).to(cpu))
                )
                A = torch.cat((A, onehot[None]), dim=0)

        if identity:
            iden = torch.eye(N).to(cpu)
            A[:N, :N] += iden
        u, s, vt = torch.linalg.svd(A)

    return u[:N, 0], A, B


def anomaly_dir_attn_cpu(blk, identity=False, bias=False, centered=False, homogeneous=False):
    """Compute attention singular defect on CPU (avoids MPS SVD issues)."""
    with torch.no_grad():
        N = blk.ls1.gamma.shape[0]

        A4 = torch.diag(blk.ls1.gamma.detach().cpu())
        A3 = blk.attn.proj.weight.detach().cpu()
        B3 = blk.attn.proj.bias.detach().cpu()
        A2 = blk.attn.qkv.weight.detach().cpu().chunk(3, dim=0)[-1]
        B2 = blk.attn.qkv.bias.detach().cpu().chunk(3, dim=0)[-1]
        A1 = torch.diag(blk.norm1.weight.detach().cpu())
        B1 = blk.norm1.bias.detach().cpu()
        A0 = (torch.eye(N) - 1 / N * torch.ones(N, N))
        A = A4 @ A3 @ A2 @ A1

        if centered:
            A = A @ A0
        B = A4 @ (A3 @ (A2 @ B1)) + A4 @ (A3 @ B2) + A4 @ B3

        if bias:
            A = torch.cat((A, B[:, None]), dim=1)
            if homogeneous:
                onehot = torch.cat((torch.zeros_like(B), torch.ones(1)))
                A = torch.cat((A, onehot[None]), dim=0)

        if identity:
            iden = torch.eye(N)
            A[:N, :N] += iden
        u, _, _ = torch.linalg.svd(A)

    return u[:N, 0], A, B


def anomaly_dir_regular(blk, homogeneous=False):
    """Combined attention + MLP anomaly direction for regular MLP blocks (CPU)."""
    _, A, b = anomaly_dir_attn_cpu(
        blk, identity=True, bias=homogeneous, centered=True, homogeneous=homogeneous
    )
    _, C, d = anomaly_dir_mlp_ls_regular(
        blk, identity=True, bias=homogeneous, bias_ls=False,
        centered=True, homogeneous=homogeneous
    )

    with torch.no_grad():
        N = b.shape[0]
        AA = C @ A
        if homogeneous:
            BB = 0
        else:
            BB = C @ b + d
        u, _, _ = torch.linalg.svd(AA)

    return u[:N, 0], AA, BB


def singular_defect_directions_regular(model):
    """Compute singular defect directions for models with regular MLP (not SwiGLU).

    All computation on CPU, results moved to model's device at the end.
    """
    device = next(model.parameters()).device
    accumulative_anomalies = []
    anomaly_dab = [anomaly_dir_regular(blk) for blk in model.blocks]
    anomaly_as = [dab[1] for dab in anomaly_dab]

    with torch.no_grad():
        aaa = torch.eye(anomaly_as[0].shape[0]).to(anomaly_as[0])
        for a in anomaly_as:
            aaa = a @ aaa
            u, _, _ = torch.linalg.svd(aaa)
            accumulative_anomalies.append(u[:, 0].to(device))
    return accumulative_anomalies


# --------------------------------------------------------------------------- #
# Cityscapes dataset for SINDER training
# --------------------------------------------------------------------------- #
class CityscapesImageDataset(Dataset):
    """Simple dataset that loads Cityscapes images for SINDER training."""

    def __init__(self, root, split="train", transform=None):
        self.transform = transform
        img_dir = Path(root) / "leftImg8bit" / split
        self.files = sorted(glob(str(img_dir / "*" / "*.png")))
        if not self.files:
            raise RuntimeError(f"No images found in {img_dir}")
        print(f"  Found {len(self.files)} {split} images")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        img = Image.open(self.files[index % len(self.files)]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


def make_transform(resolution, patch_size, center_crop=False):
    """SINDER-compatible transform."""
    MEAN = (0.485, 0.456, 0.406)
    STD = (0.229, 0.224, 0.225)
    if center_crop:
        return transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.CenterCrop(resolution),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(resolution, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN, std=STD),
            transforms.Lambda(
                lambda img: img[
                    :,
                    : (img.shape[1] - img.shape[1] % patch_size),
                    : (img.shape[2] - img.shape[2] % patch_size),
                ]
            ),
        ])


# --------------------------------------------------------------------------- #
# Neighbor loss (adapted for device-agnostic operation)
# --------------------------------------------------------------------------- #
def get_neighbor_loss_device(model, x, skip_less_than=1, mask_thr=4.0, kernel=3):
    """Compute neighbor loss, device-agnostic version of sinder.get_neighbor_loss."""
    H = x.shape[2]
    W = x.shape[3]
    x = model.prepare_tokens_with_masks(x)

    for i, blk in enumerate(model.blocks):
        x = blk(x)
        result = check_anomaly_theoretical(
            x,
            H // model.patch_size,
            W // model.patch_size,
            model.singular_defects[i],
            mask_thr=mask_thr,
            kernel=kernel,
        )
        if result is not None:
            loss_neighbor, rows, cols, T, alpha, mask_angle, x_token = result
            if len(rows) >= skip_less_than:
                assert not torch.isnan(loss_neighbor).any()
                return i, loss_neighbor, rows, cols, T, alpha, mask_angle, x_token
    return None


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def train_sinder(args):
    device = args.device
    print(f"Using device: {device}")

    # 1. Load DINOv2 ViT-B/14
    print("\n=== Loading DINOv2 ViT-B/14 ===")
    model = torch.hub.load(
        repo_or_dir=str(Path(sinder.__file__).parent.parent),
        source='local', model='dinov2_vitb14',
    )
    model.eval()
    model.interpolate_antialias = True
    model = model.to(device)

    # 2. Compute singular defect directions (adapted for regular MLP)
    print("\n=== Computing singular defect directions ===")
    print("  (Using regular MLP path for ViT-B/14)")
    model.singular_defects = singular_defect_directions_regular(model)
    print(f"  Computed {len(model.singular_defects)} defect directions")

    # 3. Prepare data
    print("\n=== Loading Cityscapes images ===")
    transform = make_transform(args.resolution, model.patch_size, center_crop=True)
    dataset = CityscapesImageDataset(args.cityscapes_root, "train", transform)

    # 4. Replace linear layers with SVD versions
    print("\n=== Replacing linear layers with SVD parameterization ===")
    model.train()
    for param in model.parameters():
        param.requires_grad = False
    replace_linear_addition_noqk(model, "model")

    # Collect trainable epsilon parameters
    epsilon_params = []
    for name, param in model.named_parameters():
        if ".epsilon" in name and param.requires_grad:
            epsilon_params.append(param)
    print(f"  Trainable epsilon parameters: {len(epsilon_params)}")
    total_trainable = sum(p.numel() for p in epsilon_params)
    print(f"  Total trainable params: {total_trainable:,}")

    optimizer = torch.optim.SGD(epsilon_params, lr=args.lr, momentum=0.9)

    # 5. Training loop
    print(f"\n=== Training ({args.max_iter} iterations) ===")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    skip_history = [False] * 1000
    best_density = 0.0

    for it in tqdm(range(args.max_iter), desc="SINDER training"):
        img = dataset[it % len(dataset)]
        H = img.shape[1] // model.patch_size
        W = img.shape[2] // model.patch_size
        density = np.array(skip_history[-1000:]).astype(float).mean()

        # Early stopping: if 75%+ images have no anomalies, we're done
        if density > 0.75 and it > 500:
            print(f"\n  Early stopping at iter {it}: skip density = {density:.2f}")
            break

        model.zero_grad()
        model.train()

        with torch.enable_grad():
            image_batch = img.unsqueeze(0).to(device)
            result = get_neighbor_loss_device(
                model, image_batch,
                skip_less_than=args.skip_less_than,
                mask_thr=args.mask_thr,
                kernel=args.kernel,
            )

        if result is None:
            skip_history.append(True)
            continue

        layer, loss, I, J, T, alpha, mask_angle, x_token = result
        skip_history.append(False)

        if torch.isnan(loss).any():
            continue

        loss.backward()

        # Zero gradients for layers far from defect
        if args.limit_layers:
            with torch.no_grad():
                for t in range(max(0, layer - args.limit_layers + 1)):
                    for p in model.blocks[t].parameters():
                        p.grad = None

        # Check for NaN gradients
        has_nan = False
        for name, param in model.named_parameters():
            if param.grad is not None and torch.isnan(param.grad).any():
                has_nan = True
                break
        if has_nan:
            continue

        optimizer.step()

        if it % 200 == 0:
            tqdm.write(
                f"  iter={it:5d}  layer={layer:2d}  loss={loss.item():.4f}  "
                f"anomalies={len(I)}  density={density:.2f}"
            )

        # Periodic checkpoint
        if it > 0 and it % 2000 == 0:
            _save_checkpoint(model, output_dir / f"sinder_vitb14_iter{it}.pth", device)

    # 6. Save final model
    print("\n=== Saving repaired model ===")
    final_path = output_dir / "sinder_vitb14.pth"
    _save_checkpoint(model, final_path, device)
    print(f"  Saved to {final_path}")
    print(f"  Final skip density: {density:.2f}")

    return final_path


def _save_checkpoint(model, path, device):
    """Convert SVD layers back to standard linear and save state_dict."""
    model.eval()
    # Need to save on CPU
    model_cpu = model.to("cpu")
    replace_back(model_cpu, "model")
    torch.save(model_cpu.state_dict(), path)
    print(f"  Checkpoint saved: {path}")
    # Restore for continued training
    model.to(device)
    replace_linear_addition_noqk(model, "model")
    # Re-enable gradients for epsilons
    for name, param in model.named_parameters():
        if ".epsilon" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False


def main():
    parser = argparse.ArgumentParser(description="Train SINDER for DINOv2 ViT-B/14")
    parser.add_argument("--cityscapes_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str,
                        default=str(PROJECT_ROOT / "refs" / "cause" / "checkpoint"))
    parser.add_argument("--device", type=str, default="auto",
                        choices=["auto", "cuda", "mps", "cpu"])
    parser.add_argument("--resolution", type=int, default=518,
                        help="Training resolution (default: 518, divisible by 14)")
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--max_iter", type=int, default=10000)
    parser.add_argument("--mask_thr", type=float, default=4.0,
                        help="Anomaly detection threshold in stds")
    parser.add_argument("--skip_less_than", type=int, default=3)
    parser.add_argument("--kernel", type=int, default=3)
    parser.add_argument("--limit_layers", type=int, default=10)
    args = parser.parse_args()

    # Auto-detect device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    train_sinder(args)


if __name__ == "__main__":
    main()
