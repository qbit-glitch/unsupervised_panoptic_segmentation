"""Train the depth-aware SGM adapter on Cityscapes pseudo-labels.

Pipeline per image:

1. Load frozen DINOv2 patch features ``(H_p, W_p, 768)`` and DepthPro depth
   ``(H, W)`` from existing caches.
2. Load (or compute and cache) depth-aware SLIC superpixels.
3. Generate the per-thing-class coarse mask supervision by running the existing
   ``depth_guided_instances`` Sobel + CC + dilation pipeline on the loaded
   semantic pseudo-label.
4. Forward the SGM adapter: fused DINO + sinusoidal depth -> per-thing-class
   foreground probability map.
5. Compute ``L = L_hard + L_soft + lambda_ad * L_ad`` and step AdamW.

Checkpoints are saved every ``--save_every`` steps so the adaptive
self-training loss has prior predictions to reference.

This trainer reuses the existing depth-CC supervision verbatim, does **not**
modify the existing ``superpixel_affinity_adapter.py`` codebase, and runs
locally on CPU/MPS/CUDA.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from mbps_pytorch.generate_depth_guided_instances import (
    DEFAULT_THING_IDS,
    WORK_H,
    WORK_W,
    depth_guided_instances,
)
from mbps_pytorch.adaptive_instance_semantics import (
    infer_semantic_spec,
    map_to_trainid,
)
from mbps_pytorch.instance_methods.depth_aware_slic import (
    DepthAwareSlicConfig,
    compute_or_load_slic,
)
from mbps_pytorch.instance_methods.rama_coarse_masks import (
    compute_or_load_coarse_masks as compute_or_load_rama_coarse_masks,
)
from mbps_pytorch.losses.superpixel_sgm import SGMLossConfig, compute_sgm_losses
from mbps_pytorch.models.instance.sgm_adapter import SGMAdapter, SGMAdapterConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


THING_TRAIN_IDS = tuple(sorted(DEFAULT_THING_IDS))  # (11, 12, 13, 14, 15, 16, 17, 18)


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class SGMDataset(Dataset):
    """One Cityscapes train image -> tensors ready for the SGM adapter.

    Caches superpixels under ``cityscapes_root/superpixels_dinodepth_slic/``.
    """

    def __init__(
        self,
        cityscapes_root: Path,
        semantic_subdir: str,
        depth_subdir: str,
        dino_subdir: str,
        split: str,
        slic_cache_subdir: str,
        slic_config: DepthAwareSlicConfig,
        max_images: int = 0,
        seed: int = 42,
        thing_grad_threshold: float = 0.05,
        thing_min_area: int = 1000,
        thing_dilation: int = 3,
        semantic_mode: str = "auto",
        centroids_path: Optional[str] = None,
        num_semantic_classes: Optional[int] = None,
        coarse_source: str = "depth_cc",  # 'depth_cc' or 'rama'
        rama_threshold: float = 0.5,
        rama_cache_subdir: str = "rama_coarse_masks",
    ) -> None:
        super().__init__()
        self.root = Path(cityscapes_root).expanduser()
        self.split = split
        self.semantic_dir = self.root / semantic_subdir / split
        self.depth_dir = self.root / depth_subdir / split
        self.dino_dir = self.root / dino_subdir / split
        self.img_dir = self.root / "leftImg8bit" / split
        self.slic_cache = self.root / slic_cache_subdir / split
        self.slic_config = slic_config
        self.thing_grad_threshold = thing_grad_threshold
        self.thing_min_area = thing_min_area
        self.thing_dilation = thing_dilation
        self.coarse_source = coarse_source
        self.rama_threshold = rama_threshold
        self.rama_cache = self.root / rama_cache_subdir / split
        # RAMA needs the cluster->trainID LUT; load eagerly if we'll use it.
        self._cluster_to_class: Optional[np.ndarray] = None
        if coarse_source == "rama":
            cents_path = (
                Path(centroids_path).expanduser() if centroids_path
                else self.root / semantic_subdir / "kmeans_centroids.npz"
            )
            data = np.load(cents_path)
            raw = data["cluster_to_class"].astype(np.uint8)
            lut = np.full(256, 255, dtype=np.uint8)
            lut[: len(raw)] = raw
            self._cluster_to_class = lut
        self.semantic_spec = infer_semantic_spec(
            self.root,
            semantic_subdir,
            semantic_mode=semantic_mode,
            centroids_path=centroids_path,
            num_semantic_classes=num_semantic_classes,
            split=split,
        )

        stems: list[tuple[str, str]] = []
        for sem_path in sorted(self.semantic_dir.rglob("*.png")):
            city = sem_path.parent.name
            stem = sem_path.stem.replace("_leftImg8bit", "")
            if (
                self._find_image(city, stem) is not None
                and self._find_depth(city, stem) is not None
                and self._find_dino(city, stem) is not None
            ):
                stems.append((city, stem))
        if max_images > 0:
            rng = random.Random(seed)
            rng.shuffle(stems)
            stems = stems[:max_images]
        self.stems = stems
        logger.info("SGMDataset(%s): %d samples", split, len(self.stems))

    def __len__(self) -> int:
        return len(self.stems)

    def _find_image(self, city: str, stem: str) -> Optional[Path]:
        for suffix in ("_leftImg8bit.png", ".png"):
            cand = self.img_dir / city / f"{stem}{suffix}"
            if cand.exists():
                return cand
        return None

    def _find_depth(self, city: str, stem: str) -> Optional[Path]:
        for suffix in (".npy", "_leftImg8bit.npy"):
            cand = self.depth_dir / city / f"{stem}{suffix}"
            if cand.exists():
                return cand
        return None

    def _find_dino(self, city: str, stem: str) -> Optional[Path]:
        for suffix in (".npy", "_leftImg8bit.npy"):
            cand = self.dino_dir / city / f"{stem}{suffix}"
            if cand.exists():
                return cand
        return None

    def _resolve_image(self, city: str, stem: str) -> Path:
        cand = self._find_image(city, stem)
        if cand is not None:
            return cand
        raise FileNotFoundError(f"image not found for {city}/{stem}")

    def _resolve_depth(self, city: str, stem: str) -> Path:
        cand = self._find_depth(city, stem)
        if cand is not None:
            return cand
        raise FileNotFoundError(f"depth not found for {city}/{stem}")

    def _resolve_dino(self, city: str, stem: str) -> Path:
        cand = self._find_dino(city, stem)
        if cand is not None:
            return cand
        raise FileNotFoundError(f"dino features not found for {city}/{stem}")

    def _resolve_semantic(self, city: str, stem: str) -> Path:
        for suffix in (".png", "_leftImg8bit.png"):
            cand = self.semantic_dir / city / f"{stem}{suffix}"
            if cand.exists():
                return cand
        raise FileNotFoundError(f"semantic not found for {city}/{stem}")

    def __getitem__(self, idx: int) -> dict:
        city, stem = self.stems[idx]
        img_path = self._resolve_image(city, stem)
        depth_path = self._resolve_depth(city, stem)
        dino_path = self._resolve_dino(city, stem)
        sem_path = self._resolve_semantic(city, stem)
        slic_path = self.slic_cache / city / f"{stem}.npy"

        image_pil = Image.open(img_path).convert("RGB")
        image_np = np.array(image_pil)
        if image_np.shape[:2] != (WORK_H, WORK_W):
            image_pil = image_pil.resize((WORK_W, WORK_H), Image.BILINEAR)
            image_np = np.array(image_pil)

        depth_np = np.load(depth_path).astype(np.float32)
        if depth_np.shape != (WORK_H, WORK_W):
            depth_np = np.array(
                Image.fromarray(depth_np).resize((WORK_W, WORK_H), Image.BILINEAR)
            ).astype(np.float32)

        semantic_full = np.array(Image.open(sem_path))
        if semantic_full.shape != (WORK_H, WORK_W):
            semantic_full = np.array(
                Image.fromarray(semantic_full).resize((WORK_W, WORK_H), Image.NEAREST)
            )
        semantic_full = map_to_trainid(semantic_full, self.semantic_spec)

        # Pre-extracted DINO features at patch resolution
        dino_np = np.load(dino_path).astype(np.float32)
        # Expect (H_p, W_p, D); some caches store (H_p*W_p, D)
        if dino_np.ndim == 2:
            side = int(np.sqrt(dino_np.shape[0]))
            if side * side != dino_np.shape[0]:
                # Fall back to nearest 2:1 ratio assumption
                hp = 32
                wp = dino_np.shape[0] // hp
                dino_np = dino_np.reshape(hp, wp, -1)
            else:
                dino_np = dino_np.reshape(side, side, -1)

        # Coarse-mask supervision: either depth-CC or RAMA MultiCut
        if self.coarse_source == "rama":
            rama_cache = self.rama_cache / city / f"{stem}.npy"
            coarse_per_class = compute_or_load_rama_coarse_masks(
                dino_path=dino_path,
                semantic_path=sem_path,
                cluster_to_class=self._cluster_to_class,
                thing_ids=THING_TRAIN_IDS,
                cache_path=rama_cache,
                out_hw=(WORK_H, WORK_W),
                threshold=self.rama_threshold,
                min_area=self.thing_min_area,
            )
        else:
            instances = depth_guided_instances(
                semantic_full,
                depth_np,
                thing_ids=set(THING_TRAIN_IDS),
                grad_threshold=self.thing_grad_threshold,
                min_area=self.thing_min_area,
                dilation_iters=self.thing_dilation,
            )
            coarse_per_class = np.zeros(
                (len(THING_TRAIN_IDS), WORK_H, WORK_W), dtype=np.uint8
            )
            cls_to_idx = {c: i for i, c in enumerate(THING_TRAIN_IDS)}
            for mask, cls, _score in instances:
                if cls in cls_to_idx:
                    coarse_per_class[cls_to_idx[cls]] |= mask.astype(np.uint8)

        # Superpixels (cached)
        sp_labels = compute_or_load_slic(
            img_path, depth_path, dino_path, slic_path, self.slic_config
        )

        # Upsample DINO features once for the loss (D, H, W)
        dino_full = torch.from_numpy(dino_np).permute(2, 0, 1).unsqueeze(0)
        dino_full = F.interpolate(dino_full, size=(WORK_H, WORK_W), mode="bilinear",
                                  align_corners=False).squeeze(0)

        sample = {
            "image": torch.from_numpy(image_np).permute(2, 0, 1).float() / 255.0,  # (3, H, W)
            "depth": torch.from_numpy(depth_np),                                   # (H, W)
            "dino_full": dino_full,                                                # (D, H, W)
            "dino_patch": torch.from_numpy(dino_np.reshape(-1, dino_np.shape[-1])),  # (N, D)
            "sp_labels": torch.from_numpy(sp_labels.astype(np.int64)),             # (H, W)
            "coarse": torch.from_numpy(coarse_per_class),                          # (T, H, W)
            "stem": stem,
            "city": city,
        }
        return sample


def _identity_collate(batch: List[dict]) -> List[dict]:
    return batch


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def _patch_depth_from_full(depth: torch.Tensor, h_p: int, w_p: int) -> torch.Tensor:
    """Average-pool the full-resolution depth map to the DINO patch grid."""
    d = depth.unsqueeze(0).unsqueeze(0)
    d = F.adaptive_avg_pool2d(d, (h_p, w_p))
    return d.squeeze(0).squeeze(0).reshape(-1)


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    slic_config = DepthAwareSlicConfig(
        n_segments=args.n_segments,
        compactness=args.slic_compactness,
        alpha_slic=args.slic_alpha_depth,
        beta_slic=args.slic_beta_dino,
    )
    loss_config = SGMLossConfig(
        alpha_1=args.alpha_color,
        alpha_d=args.alpha_depth,
        alpha_2=args.alpha_mst,
        lambda_d=args.lambda_depth,
        lambda_f=args.lambda_dino,
        lambda_ad=args.lambda_ad,
    )
    adapter_config = SGMAdapterConfig(
        d_dino=args.d_dino,
        n_fusion_layers=args.n_fusion_layers,
        n_heads=args.n_heads,
        n_thing=len(THING_TRAIN_IDS),
        patch_h=args.patch_h,
        patch_w=args.patch_w,
        use_concat_fusion=args.fusion == "concat",
    )

    train_set = SGMDataset(
        cityscapes_root=Path(args.cityscapes_root),
        semantic_subdir=args.semantic_subdir,
        depth_subdir=args.depth_subdir,
        dino_subdir=args.dino_subdir,
        split="train",
        slic_cache_subdir=args.slic_cache_subdir,
        slic_config=slic_config,
        max_images=args.max_train_images,
        seed=args.seed,
        thing_grad_threshold=args.coarse_tau,
        thing_min_area=args.coarse_min_area,
        thing_dilation=args.coarse_dilation,
        semantic_mode=args.semantic_mode,
        centroids_path=args.centroids_path,
        num_semantic_classes=args.num_semantic_classes,
        coarse_source=args.coarse_source,
        rama_threshold=args.rama_threshold,
        rama_cache_subdir=args.rama_cache_subdir,
    )
    loader = DataLoader(
        train_set, batch_size=1, shuffle=True,
        num_workers=args.num_workers, collate_fn=_identity_collate,
    )

    adapter = SGMAdapter(adapter_config).to(device)
    n_param = adapter.num_parameters()
    logger.info("SGMAdapter trainable params: %d (~%.2f M)", n_param, n_param / 1e6)

    opt = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=args.wd)
    out_dir = Path(args.output_dir).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps({
        "slic": asdict(slic_config),
        "loss": asdict(loss_config),
        "adapter": asdict(adapter_config),
        "args": vars(args),
        "thing_train_ids": list(THING_TRAIN_IDS),
    }, indent=2))

    checkpoint_buffer: list[torch.Tensor] = []
    step = 0
    t0 = time.time()
    total_steps = args.max_steps if args.max_steps > 0 else args.epochs * len(loader)
    pbar = tqdm(
        total=total_steps,
        desc="sgm_adapter",
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=1.0,
        smoothing=0.3,
        ascii=True,
    )
    for epoch in range(args.epochs):
        for batch in loader:
            sample = batch[0]
            image = sample["image"].to(device)
            depth = sample["depth"].to(device)
            dino_full = sample["dino_full"].to(device)
            dino_patch = sample["dino_patch"].to(device).unsqueeze(0)        # (1, N, D)
            sp_labels = sample["sp_labels"].to(device)
            coarse = sample["coarse"].to(device).float()                     # (T, H, W)

            depth_patch = _patch_depth_from_full(depth, args.patch_h, args.patch_w).unsqueeze(0)
            m_tilde = adapter(dino_patch, depth_patch, out_hw=(WORK_H, WORK_W)).squeeze(0)

            losses = compute_sgm_losses(
                m_tilde=m_tilde,
                coarse_mask=coarse,
                image=image,
                depth=depth,
                dino_full=dino_full,
                sp_labels=sp_labels,
                checkpoints=checkpoint_buffer,
                config=loss_config,
            )
            loss = losses["loss"]
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(adapter.parameters(), max_norm=1.0)
            opt.step()

            l_ad_val = float(losses["L_ad"]) if isinstance(losses["L_ad"], torch.Tensor) else 0.0
            pbar.set_postfix({
                "ep": epoch,
                "loss": f"{float(loss.detach()):.4f}",
                "Lh": f"{float(losses['L_hard']):.4f}",
                "Ls": f"{float(losses['L_soft']):.4f}",
                "Lad": f"{l_ad_val:.4f}",
                "n_sp": int(sp_labels.max().item()) + 1,
            })
            pbar.update(1)

            if args.log_every > 0 and step % args.log_every == 0:
                # Permanent record in the log file at coarse intervals
                logger.info(
                    "epoch=%d step=%d loss=%.4f L_hard=%.4f L_soft=%.4f L_ad=%.4f n_sp=%d",
                    epoch, step, float(loss.detach()),
                    float(losses["L_hard"]), float(losses["L_soft"]),
                    l_ad_val, int(sp_labels.max().item()) + 1,
                )

            if args.save_every > 0 and step > 0 and step % args.save_every == 0:
                ckpt_path = out_dir / f"adapter_step{step:06d}.pt"
                torch.save({
                    "state_dict": adapter.state_dict(),
                    "step": step,
                    "adapter_config": asdict(adapter_config),
                }, ckpt_path)
                logger.info("saved checkpoint: %s", ckpt_path)
                if args.adaptive_pred_buffer > 0:
                    checkpoint_buffer.append(m_tilde.detach().cpu())
                    if len(checkpoint_buffer) > args.adaptive_pred_buffer:
                        checkpoint_buffer.pop(0)

            step += 1
            if args.max_steps > 0 and step >= args.max_steps:
                logger.info("hit max_steps=%d, stopping", args.max_steps)
                break
        if args.max_steps > 0 and step >= args.max_steps:
            break
    pbar.close()

    final_path = out_dir / "adapter_final.pt"
    torch.save({
        "state_dict": adapter.state_dict(),
        "step": step,
        "adapter_config": asdict(adapter_config),
    }, final_path)
    logger.info("done in %.1fs; final checkpoint: %s", time.time() - t0, final_path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Train depth-aware SGM adapter")
    p.add_argument("--cityscapes_root", type=str, required=True)
    p.add_argument("--semantic_subdir", type=str, default="pseudo_semantic_raw_k80")
    p.add_argument("--semantic_mode", type=str, default="auto",
                   choices=("auto", "cluster", "cause27", "trainid"))
    p.add_argument("--centroids_path", type=str, default=None)
    p.add_argument("--num_semantic_classes", type=int, default=None)
    p.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    p.add_argument("--dino_subdir", type=str, default="dinov2_features")
    p.add_argument("--slic_cache_subdir", type=str, default="superpixels_dinodepth_slic")
    p.add_argument("--output_dir", type=str, default="checkpoints/sgm_adapter")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--max_steps", type=int, default=0)
    p.add_argument("--max_train_images", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=500)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_workers", type=int, default=0)
    # Adapter architecture
    p.add_argument("--fusion", type=str, choices=("xattn", "concat"), default="xattn")
    p.add_argument("--d_dino", type=int, default=768)
    p.add_argument("--n_fusion_layers", type=int, default=3)
    p.add_argument("--n_heads", type=int, default=8)
    p.add_argument("--patch_h", type=int, default=32)
    p.add_argument("--patch_w", type=int, default=64)
    # SLIC
    p.add_argument("--n_segments", type=int, default=1500)
    p.add_argument("--slic_compactness", type=float, default=10.0)
    p.add_argument("--slic_alpha_depth", type=float, default=1.0)
    p.add_argument("--slic_beta_dino", type=float, default=0.5)
    # Loss
    p.add_argument("--alpha_color", type=float, default=0.05)
    p.add_argument("--alpha_depth", type=float, default=0.10)
    p.add_argument("--alpha_mst", type=float, default=0.30)
    p.add_argument("--lambda_depth", type=float, default=5.0)
    p.add_argument("--lambda_dino", type=float, default=1.0)
    p.add_argument("--lambda_ad", type=float, default=1.0)
    p.add_argument("--adaptive_pred_buffer", type=int, default=4)
    # Coarse mask
    p.add_argument("--coarse_source", type=str, choices=("depth_cc", "rama"), default="depth_cc",
                   help="Source of coarse-mask supervision: depth-CC pipeline or RAMA MultiCut.")
    p.add_argument("--coarse_tau", type=float, default=0.05,
                   help="depth_cc only: Sobel gradient threshold")
    p.add_argument("--coarse_min_area", type=int, default=1000)
    p.add_argument("--coarse_dilation", type=int, default=3,
                   help="depth_cc only: dilation iterations")
    p.add_argument("--rama_threshold", type=float, default=0.5,
                   help="rama only: cosine-similarity threshold for affinity costs")
    p.add_argument("--rama_cache_subdir", type=str, default="rama_coarse_masks",
                   help="rama only: cache subdir under cityscapes_root")
    return p


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
