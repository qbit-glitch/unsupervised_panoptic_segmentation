#!/usr/bin/env python3
"""Build M-of-N rare-pixel protected mask using k=80-only sources.

Sources (all unsupervised, all k=80 clustering only):
  S1: DCFA k=80 cluster ID differs from raw-DINOv3 k=80 cluster ID at this pixel
      (DCFA's adapter relocates this pixel to a different cluster than the raw
       DINOv3 features would; signals a region DCFA "thinks differently" about)
  S3: Sobel depth edge inside a thin (<max_band_width=20 px) horizontal run
      (captures fine vertical structures: guard rails, poles, fence posts)
  S4: Small connected component within a k=80 cluster (CC < 200 px AND
       ≥ 2 px disconnected from the cluster's main mass)
      (a k=80 cluster usually has 1–2 dominant CCs per image; isolated
       satellites are likely rare-class fragments wrongly merged into a
       dominant cluster)

A pixel is rare-protected if M of N sources agree (default M=2, N=3).

Optional flags allow adding:
  --include_s2  Step A would-overwrite (replays SIMCF Step A logic dry-run;
                higher recall but correlates with CN-SIMCF Step A by construction)
  --include_s5  Multi-seed DCFA disagreement (requires --extra_dcfa_dirs)

Output: per-image binary mask at full image resolution, saved as PNG.
"""
import argparse
import logging
import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

NUM_CLASSES = 19
MASK_H, MASK_W = 32, 64


# ---------------------------------------------------------------------------
# Sources (all k=80-compatible)
# ---------------------------------------------------------------------------

def source_s1_dcfa_disagreement(dcfa_sem: np.ndarray, raw_sem: np.ndarray) -> np.ndarray:
    """S1: DCFA cluster differs from raw-DINOv3 cluster at this pixel."""
    if dcfa_sem.shape != raw_sem.shape:
        raw_sem = np.array(
            Image.fromarray(raw_sem).resize(
                (dcfa_sem.shape[1], dcfa_sem.shape[0]), Image.NEAREST)
        )
    return dcfa_sem != raw_sem


def source_s2_step_a_would_overwrite(semantic: np.ndarray, instance: np.ndarray,
                                      cluster_to_class: np.ndarray,
                                      num_clusters: int = 80) -> np.ndarray:
    """S2 (optional): replays SIMCF Step A and returns mask of pixels that
    would be reassigned. Note: correlates with CN-SIMCF Step A by construction.
    """
    out = np.zeros_like(semantic, dtype=bool)
    for iid in np.unique(instance):
        if iid == 0:
            continue
        ys, xs = np.where(instance == iid)
        if len(ys) == 0:
            continue
        clusters = semantic[ys, xs]
        train_ids = cluster_to_class[clusters]
        valid_tids = train_ids[train_ids < NUM_CLASSES]
        if len(valid_tids) == 0:
            continue
        majority_tid = int(np.bincount(valid_tids, minlength=NUM_CLASSES).argmax())
        inconsistent = (train_ids != majority_tid) & (train_ids < NUM_CLASSES)
        if inconsistent.any():
            out[ys[inconsistent], xs[inconsistent]] = True
    return out


def source_s3_depth_thin_band(depth: np.ndarray, target_shape: tuple,
                               edge_thresh: float = 0.20,
                               max_band_width: int = 20) -> np.ndarray:
    """S3: depth-edge pixels inside a thin (<max_band_width px) horizontal run."""
    if depth.shape != target_shape:
        depth = np.array(
            Image.fromarray(depth.astype(np.float32)).resize(
                (target_shape[1], target_shape[0]), Image.BILINEAR)
        )
    sx = ndimage.sobel(depth, axis=0)
    sy = ndimage.sobel(depth, axis=1)
    grad = np.sqrt(sx ** 2 + sy ** 2)
    grad_norm = grad / (grad.max() + 1e-8)
    edges = grad_norm > edge_thresh

    out = np.zeros_like(edges)
    for i in range(edges.shape[0]):
        row = edges[i]
        labeled, n = ndimage.label(row)
        for k in range(1, n + 1):
            run = labeled == k
            run_size = int(run.sum())
            if 0 < run_size < max_band_width:
                out[i, run] = True
    return out


def source_s4_small_cc_within_cluster(semantic: np.ndarray,
                                       max_cc_size: int = 200,
                                       min_separation_px: int = 2,
                                       num_clusters: int = 80) -> np.ndarray:
    """S4: small CCs (<max_cc_size px) inside a k=80 cluster, disconnected
    from the cluster's main CC by at least min_separation_px.
    """
    out = np.zeros_like(semantic, dtype=bool)
    for cl in range(num_clusters):
        cluster_mask = semantic == cl
        if not cluster_mask.any():
            continue
        labeled, n_cc = ndimage.label(cluster_mask)
        if n_cc <= 1:
            continue
        cc_sizes = ndimage.sum(cluster_mask, labeled, range(1, n_cc + 1))
        main_cc_idx = int(np.argmax(cc_sizes)) + 1
        # All small CCs (not the main one)
        for cc_id in range(1, n_cc + 1):
            if cc_id == main_cc_idx:
                continue
            cc = labeled == cc_id
            cc_size = int(cc.sum())
            if cc_size >= max_cc_size:
                continue
            # Check separation from main CC
            main_mask = labeled == main_cc_idx
            main_dilated = ndimage.binary_dilation(
                main_mask, iterations=min_separation_px)
            if (cc & main_dilated).any():
                continue  # touches main CC, skip
            out |= cc
    return out


def source_s5_multi_seed_disagreement(seeds: list[np.ndarray]) -> np.ndarray:
    """S5 (optional): multi-seed DCFA disagreement.

    Returns pixels where ≥ 1 seed disagrees with the modal cluster across seeds.
    """
    if len(seeds) < 2:
        return np.zeros_like(seeds[0], dtype=bool)
    target = seeds[0].shape
    aligned = [s if s.shape == target else
               np.array(Image.fromarray(s).resize((target[1], target[0]), Image.NEAREST))
               for s in seeds]
    stack = np.stack(aligned, axis=0)  # (N_seeds, H, W)
    # Mode across seeds (per pixel)
    out = np.zeros(target, dtype=bool)
    # Vectorized mode is tricky for arbitrary values; use a loop over unique values
    for s in aligned:
        out |= (s != aligned[0])
    return out


# ---------------------------------------------------------------------------
# Path resolution
# ---------------------------------------------------------------------------

def _resolve_path(base: Path, split: str, city: str, stem: str, ext: str) -> Path | None:
    base_stem = stem.replace("_leftImg8bit", "")
    candidates = [
        base / split / city / f"{stem}.{ext}",
        base / split / city / f"{base_stem}.{ext}",
        base / split / city / f"{base_stem}_leftImg8bit.{ext}",
        base / city / f"{stem}.{ext}",
        base / f"{stem}.{ext}",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _resolve_cups_pair(base: Path, split: str, city: str, stem: str) -> tuple[Path, Path] | tuple[None, None]:
    """Resolve flat or split/city CUPS semantic+instance files.

    City-structured semantic roots use stems such as ``aachen_000000_000019``;
    CUPS flat pseudo-label roots usually append ``_leftImg8bit`` before the
    ``_semantic.png`` / ``_instance.png`` suffixes. Accept both forms.
    """
    stem_variants = [stem]
    if stem.endswith("_leftImg8bit"):
        stem_variants.append(stem.replace("_leftImg8bit", ""))
    else:
        stem_variants.append(f"{stem}_leftImg8bit")

    root_variants = [
        base,
        base / split / city,
        base / city,
    ]
    for root in root_variants:
        for s in stem_variants:
            sem = root / f"{s}_semantic.png"
            inst = root / f"{s}_instance.png"
            if sem.exists() and inst.exists():
                return sem, inst
    return None, None


def _extract_city(stem: str) -> str:
    m = re.match(r"^(.+?)_\d{6}_\d{6}_leftImg8bit$", stem)
    if m:
        return m.group(1)
    return stem.split("_")[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cityscapes_root", required=True)
    parser.add_argument("--dcfa_pseudo_dir", required=True,
                        help="DCFA k=80 semantic pseudo-labels root (<split>/<city>/*.png)")
    parser.add_argument("--raw_pseudo_dir", required=True,
                        help="Raw DINOv3 k=80 semantic pseudo-labels root (no DCFA)")
    parser.add_argument("--depth_dir", required=True,
                        help="DepthPro depth maps root")
    parser.add_argument("--cn_simcf_input_dir", required=True,
                        help="CUPS-format dir with semantic.png + instance.png "
                             "(used for S2 and S4 — semantic is k=80 cluster IDs)")
    parser.add_argument("--centroids_path", required=True,
                        help="kmeans_centroids.npz (for cluster_to_class in S2)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--num_clusters", type=int, default=80)
    parser.add_argument("--edge_thresh", type=float, default=0.20)
    parser.add_argument("--max_band_width", type=int, default=20)
    parser.add_argument("--max_cc_size", type=int, default=200)
    parser.add_argument("--min_separation_px", type=int, default=2)
    parser.add_argument("--include_s2", action="store_true",
                        help="Add Step A would-overwrite source (correlates with CN-SIMCF Step A)")
    parser.add_argument("--include_s5", action="store_true",
                        help="Add multi-seed DCFA disagreement (requires --extra_dcfa_dirs)")
    parser.add_argument("--extra_dcfa_dirs", nargs="*", default=[],
                        help="Extra DCFA seed pseudo-label dirs for S5")
    parser.add_argument("--agreement_required", type=int, default=2,
                        help="Number of sources required to mark a pixel protected")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser() / args.split
    output_dir.mkdir(parents=True, exist_ok=True)

    dcfa_root = Path(args.dcfa_pseudo_dir).expanduser()
    raw_root = Path(args.raw_pseudo_dir).expanduser()
    depth_root = Path(args.depth_dir).expanduser()
    cn_input_root = Path(args.cn_simcf_input_dir).expanduser()
    extra_dcfa_roots = [Path(d).expanduser() for d in args.extra_dcfa_dirs]

    centroid_data = np.load(args.centroids_path)
    raw_c2c = centroid_data["cluster_to_class"].astype(np.uint8)
    cluster_to_class = np.full(256, 255, dtype=np.uint8)
    cluster_to_class[:len(raw_c2c)] = raw_c2c

    # Iterate by DCFA dir cities
    dcfa_split_dir = dcfa_root / args.split
    if not dcfa_split_dir.exists():
        dcfa_split_dir = dcfa_root
    cities = sorted([c.name for c in dcfa_split_dir.iterdir() if c.is_dir()])
    logger.info(f"Cities: {cities}")
    logger.info(f"Sources active: S1=YES, S3=YES, S4=YES, "
                f"S2={'YES' if args.include_s2 else 'NO'}, "
                f"S5={'YES' if args.include_s5 else 'NO'}")
    n_sources = 3 + int(args.include_s2) + int(args.include_s5)
    logger.info(f"Agreement required: {args.agreement_required} of {n_sources}")

    n_done = 0
    n_skipped = 0
    total_protected_px = 0
    total_px = 0

    for city in cities:
        out_city = output_dir / city
        out_city.mkdir(parents=True, exist_ok=True)
        city_dir = dcfa_split_dir / city
        for dcfa_path in tqdm(sorted(city_dir.glob("*.png")), desc=city):
            stem = dcfa_path.stem
            try:
                dcfa = np.array(Image.open(dcfa_path))
                raw_path = _resolve_path(raw_root, args.split, city, stem, "png")
                depth_path = _resolve_path(depth_root, args.split, city, stem, "npy")

                cn_sem_path, cn_inst_path = _resolve_cups_pair(
                    cn_input_root, args.split, city, stem
                )

                if raw_path is None or depth_path is None or cn_sem_path is None \
                        or cn_inst_path is None or not cn_sem_path.exists() \
                        or not cn_inst_path.exists():
                    n_skipped += 1
                    continue

                raw = np.array(Image.open(raw_path))
                depth = np.load(str(depth_path)).astype(np.float32)
                cn_sem = np.array(Image.open(cn_sem_path))
                cn_inst = np.array(Image.open(cn_inst_path))
            except (FileNotFoundError, OSError) as e:
                logger.warning(f"Missing source for {stem}: {e}")
                n_skipped += 1
                continue

            full_shape = dcfa.shape
            target_shape = (MASK_H, MASK_W)

            # Compute sources at the intended DINO patch grid, then upsample.
            dcfa_s = np.array(
                Image.fromarray(dcfa).resize((MASK_W, MASK_H), Image.NEAREST)
            )
            raw_s = np.array(
                Image.fromarray(raw).resize((MASK_W, MASK_H), Image.NEAREST)
            )

            s1 = source_s1_dcfa_disagreement(dcfa_s, raw_s)
            s3 = source_s3_depth_thin_band(depth, target_shape,
                                           args.edge_thresh, args.max_band_width)

            # S4 needs k=80 cluster IDs; cn_sem holds them
            if cn_sem.shape != target_shape:
                cn_sem_resized = np.array(
                    Image.fromarray(cn_sem).resize(
                            (MASK_W, MASK_H), Image.NEAREST)
                )
            else:
                cn_sem_resized = cn_sem
            s4 = source_s4_small_cc_within_cluster(
                cn_sem_resized, args.max_cc_size, args.min_separation_px,
                args.num_clusters
            )

            sources = [s1, s3, s4]

            if args.include_s2:
                if cn_inst.shape != target_shape:
                    cn_inst_resized = np.array(
                        Image.fromarray(cn_inst).resize(
                            (MASK_W, MASK_H), Image.NEAREST)
                    )
                else:
                    cn_inst_resized = cn_inst
                s2 = source_s2_step_a_would_overwrite(
                    cn_sem_resized, cn_inst_resized, cluster_to_class,
                    args.num_clusters
                )
                sources.append(s2)

            if args.include_s5 and extra_dcfa_roots:
                seeds_arrays = [dcfa_s]
                for extra_root in extra_dcfa_roots:
                    extra_path = _resolve_path(extra_root, args.split, city, stem, "png")
                    if extra_path is not None:
                        extra = np.array(Image.open(extra_path))
                        extra_s = np.array(
                            Image.fromarray(extra).resize((MASK_W, MASK_H), Image.NEAREST)
                        )
                        seeds_arrays.append(extra_s)
                if len(seeds_arrays) >= 2:
                    s5 = source_s5_multi_seed_disagreement(seeds_arrays)
                    sources.append(s5)

            agreement = np.zeros(target_shape, dtype=np.uint8)
            for s in sources:
                agreement += s.astype(np.uint8)
            protected_small = agreement >= args.agreement_required
            protected = np.array(
                Image.fromarray(protected_small.astype(np.uint8) * 255)
                .resize((full_shape[1], full_shape[0]), Image.NEAREST)
            ) > 0

            total_protected_px += int(protected.sum())
            total_px += protected.size

            Image.fromarray(protected.astype(np.uint8) * 255).save(
                str(out_city / f"{stem}.png")
            )
            n_done += 1

    pct = 100 * total_protected_px / max(total_px, 1)
    logger.info(f"\n{'='*60}")
    logger.info(f"Wrote {n_done} masks (skipped {n_skipped})")
    logger.info(f"Protected pixels: {total_protected_px}/{total_px} ({pct:.2f}%)")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    main()
