"""Diagnose why L_soft saturates at 0 in the SGM trainer.

Dumps for a single Cityscapes train image:

* Distribution of per-superpixel ``P_k`` (per class).
* Distribution of MST-propagated ``hat_P_k`` (per class).
* Element-wise ``|P_k - hat_P_k|`` and the resulting ``L_soft`` term.
* Sweep of ``alpha_mst`` to see how the propagation sharpness affects the
  spread between ``P`` and ``hat_P``.

Run with the same SLIC cache used during training to get apples-to-apples
numbers.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from mbps_pytorch.generate_depth_guided_instances import WORK_H, WORK_W
from mbps_pytorch.instance_methods.depth_aware_slic import (
    DepthAwareSlicConfig, compute_or_load_slic,
)
from mbps_pytorch.losses.superpixel_sgm import (
    SGMLossConfig, compute_sp_means, depth_aware_edge_weights, mst_soft_labels,
    pixel_to_sp_weight, superpixel_adjacency, superpixel_foreground_prob,
)
from mbps_pytorch.models.instance.sgm_adapter import SGMAdapter, SGMAdapterConfig

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _load_adapter(ckpt_path: Path, device: torch.device) -> SGMAdapter:
    state = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    cfg = SGMAdapterConfig(**state.get("adapter_config", {}))
    a = SGMAdapter(cfg).to(device).eval()
    a.load_state_dict(state["state_dict"])
    return a


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--cityscapes_root", type=str, required=True)
    p.add_argument("--depth_subdir", type=str, default="depth_depthpro")
    p.add_argument("--dino_subdir", type=str, default="dinov2_features")
    p.add_argument("--slic_cache_subdir", type=str, default="superpixels_dinodepth_slic")
    p.add_argument("--city", type=str, default="aachen")
    p.add_argument("--stem", type=str, default="aachen_000000_000019")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--n_segments", type=int, default=400)
    p.add_argument("--alpha_mst_sweep", type=str, default="0.30,0.10,0.05,0.03,0.01")
    args = p.parse_args()

    device = torch.device(args.device)
    cs = Path(args.cityscapes_root).expanduser()
    img_path = cs / "leftImg8bit/train" / args.city / f"{args.stem}_leftImg8bit.png"
    depth_path = cs / args.depth_subdir / "train" / args.city / f"{args.stem}.npy"
    if not depth_path.exists():
        depth_path = cs / args.depth_subdir / "train" / args.city / f"{args.stem}_leftImg8bit.npy"
    dino_path = cs / args.dino_subdir / "train" / args.city / f"{args.stem}_leftImg8bit.npy"
    if not dino_path.exists():
        dino_path = cs / args.dino_subdir / "train" / args.city / f"{args.stem}.npy"
    slic_path = cs / args.slic_cache_subdir / "train" / args.city / f"{args.stem}.npy"

    # ----- Load adapter and run forward -----
    adapter = _load_adapter(Path(args.checkpoint).expanduser(), device)
    image_np = np.array(Image.open(img_path).convert("RGB"))
    if image_np.shape[:2] != (WORK_H, WORK_W):
        image_np = np.array(
            Image.open(img_path).convert("RGB").resize((WORK_W, WORK_H), Image.BILINEAR)
        )
    depth_np = np.load(depth_path).astype(np.float32)
    if depth_np.shape != (WORK_H, WORK_W):
        depth_np = np.array(
            Image.fromarray(depth_np).resize((WORK_W, WORK_H), Image.BILINEAR)
        ).astype(np.float32)
    dino_np = np.load(dino_path).astype(np.float32)
    if dino_np.ndim == 2:
        dino_np = dino_np.reshape(32, 64, -1)

    # SLIC: use existing cache if present
    sp_labels_np = compute_or_load_slic(
        img_path, depth_path, dino_path, slic_path,
        DepthAwareSlicConfig(n_segments=args.n_segments),
    )
    image_t = torch.from_numpy(image_np).permute(2, 0, 1).float().div(255).to(device)
    depth_t = torch.from_numpy(depth_np).to(device)
    sp_t = torch.from_numpy(sp_labels_np.astype(np.int64)).to(device)
    dino_full = F.interpolate(
        torch.from_numpy(dino_np).permute(2, 0, 1)[None].to(device),
        size=(WORK_H, WORK_W), mode="bilinear", align_corners=False,
    ).squeeze(0)

    # Adapter forward
    dino_patch_flat = torch.from_numpy(dino_np.reshape(-1, dino_np.shape[-1]))[None].to(device)
    depth_patch = F.adaptive_avg_pool2d(
        depth_t[None, None], (32, 64)
    ).reshape(-1)[None]
    with torch.no_grad():
        m_tilde = adapter(dino_patch_flat, depth_patch, out_hw=(WORK_H, WORK_W)).squeeze(0)
    logger.info("m_tilde stats: min=%.4f max=%.4f mean=%.4f frac>0.5=%.3f",
                float(m_tilde.min()), float(m_tilde.max()),
                float(m_tilde.mean()), float((m_tilde > 0.5).float().mean()))
    # Per-class histogram of m_tilde
    for c in range(m_tilde.shape[0]):
        vals = m_tilde[c].flatten()
        bins = torch.histc(vals, bins=10, min=0.0, max=1.0).cpu().tolist()
        logger.info("  m_tilde[c=%d] histogram (bins 0..1): %s",
                    c, [int(b) for b in bins])

    # ----- Compute SP means, edges -----
    k = int(sp_t.max().item()) + 1
    mu_c, bar_d, bar_f = compute_sp_means(image_t, depth_t, dino_full, sp_t, k)
    edges = superpixel_adjacency(sp_t)
    base_cfg = SGMLossConfig()
    weights = depth_aware_edge_weights(edges, mu_c, bar_d, bar_f, base_cfg)
    logger.info("|S|=%d edges=%d w stats: min=%.4f max=%.4f mean=%.4f",
                k, edges.shape[0],
                float(weights.min()), float(weights.max()), float(weights.mean()))

    # ----- P_k and hat_P_k for class 0 (person) -----
    delta = pixel_to_sp_weight(image_t, depth_t, sp_t, mu_c, bar_d, base_cfg)
    print("\n=== alpha_mst sweep ===")
    print(f"{'alpha_mst':>10}  {'class':>5}  {'P mean':>8}  {'P std':>8}  "
          f"{'hatP mean':>10}  {'hatP std':>9}  {'L1(P,hatP)':>11}")
    for am in [float(x) for x in args.alpha_mst_sweep.split(",")]:
        cfg = SGMLossConfig(alpha_2=am)
        for c in range(m_tilde.shape[0]):
            P = superpixel_foreground_prob(m_tilde[c], delta, sp_t, k)
            hat_P = mst_soft_labels(P, edges, weights, cfg)
            l1 = float((P - hat_P).abs().mean())
            print(f"  {am:9.3f}  {c:5d}  {float(P.mean()):8.5f}  "
                  f"{float(P.std()):8.5f}  {float(hat_P.mean()):10.5f}  "
                  f"{float(hat_P.std()):9.5f}  {l1:11.5f}")


if __name__ == "__main__":
    main()
