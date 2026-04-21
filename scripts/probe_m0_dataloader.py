"""Non-invasive dataloader probe for Stage-2 M0 (Mask2Former + ViT-Adapter).

Measures thing/stuff instance distribution per crop on the exact
PseudoLabelDataset that M0 training consumes, so we can confirm or rule
out hypothesis D (PQ_things=0 driven by crops with zero thing instances,
not by model/loss issues).

Runs CPU-only, reads the same pseudo labels + augmentations as train.py
lines 121-138. Run in a second terminal while M0 training is live on the
GPU — does not touch the training process.

Usage (from repo root on anydesk):
    python3 scripts/probe_m0_dataloader.py \
        --config configs/stage2_m2f/M0_baseline_dinov3_vitb_k80_anydesk.yaml \
        --num_samples 200
"""
from __future__ import annotations

import argparse
import logging
import statistics
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

import torch

# Ensure refs/cups is on the path (train.py lives there).
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "refs" / "cups"))

from cups.augmentation import get_pseudo_label_augmentations  # noqa: E402
from cups.config import get_default_config  # noqa: E402
from cups.data import PseudoLabelDataset  # noqa: E402

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger("probe_m0")


def build_training_dataset(config) -> PseudoLabelDataset:
    """Mirror refs/cups/train.py:121-138 exactly."""
    return PseudoLabelDataset(
        root=config.DATA.ROOT,
        root_pseudo=config.DATA.ROOT_PSEUDO,
        return_detectron2_format=True,
        ground_truth_scale=config.DATA.SCALE,
        crop_resolution=config.DATA.CROP_RESOLUTION,
        thing_stuff_threshold=config.DATA.THING_STUFF_THRESHOLD,
        ignore_unknown_thing_regions=config.DATA.IGNORE_UNKNOWN_THING_REGIONS,
        augmentations=get_pseudo_label_augmentations(config.DATA.CROP_RESOLUTION),
        dataset=config.DATA.DATASET,
        only_use_non_empty_samples=True,
        depth_subdir=getattr(config.DATA, "DEPTH_SUBDIR", ""),
        num_pseudo_classes=config.DATA.NUM_PSEUDO_CLASSES,
        load_pseudo_onehot=(
            getattr(config.MODEL.SEM_SEG_HEAD, "STUFF_KD_WEIGHT", 0.0) > 0
        ),
    )


def count_stuff_labels(sem_seg: torch.Tensor) -> List[int]:
    """Return unique stuff class ids in a crop, excluding void (0) and ignore (255)."""
    u = torch.unique(sem_seg)
    u = u[(u != 0) & (u != 255)]
    return u.tolist()


def summarise(name: str, values: List[int]) -> str:
    if not values:
        return f"{name}: (empty)"
    return (
        f"{name}: mean={statistics.mean(values):.2f} "
        f"median={statistics.median(values):.1f} "
        f"min={min(values)} max={max(values)} "
        f"stdev={statistics.pstdev(values):.2f}"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--num_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    torch.manual_seed(args.seed)

    log.info(f"Loading config: {args.config}")
    cfg = get_default_config(experiment_config_file=args.config)
    log.info(
        f"ROOT={cfg.DATA.ROOT}\nROOT_PSEUDO={cfg.DATA.ROOT_PSEUDO}\n"
        f"CROP_RESOLUTION={cfg.DATA.CROP_RESOLUTION}\n"
        f"THING_STUFF_THRESHOLD={cfg.DATA.THING_STUFF_THRESHOLD}\n"
        f"IGNORE_UNKNOWN_THING_REGIONS={cfg.DATA.IGNORE_UNKNOWN_THING_REGIONS}\n"
        f"NUM_PSEUDO_CLASSES={cfg.DATA.NUM_PSEUDO_CLASSES}\n"
    )

    dataset = build_training_dataset(cfg)
    dataset_size = len(dataset)
    n_probe = min(args.num_samples, dataset_size)
    log.info(f"Dataset length: {dataset_size}  |  probing {n_probe} samples")

    # Random sample with replacement-like behaviour: shuffle then take first n.
    rng = torch.Generator().manual_seed(args.seed)
    order = torch.randperm(dataset_size, generator=rng).tolist()[:n_probe]

    thing_counts: List[int] = []
    stuff_counts: List[int] = []
    thing_class_counter: Counter = Counter()
    stuff_class_counter: Counter = Counter()
    zero_thing_samples = 0

    for i, idx in enumerate(order):
        sample: Dict = dataset[idx]
        instances = sample["instances"]
        sem_seg = sample["sem_seg"]

        n_thing = int(len(instances.gt_classes))
        thing_counts.append(n_thing)
        if n_thing == 0:
            zero_thing_samples += 1
        for cid in instances.gt_classes.tolist():
            thing_class_counter[int(cid)] += 1

        stuff_ids = count_stuff_labels(sem_seg)
        stuff_counts.append(len(stuff_ids))
        for cid in stuff_ids:
            stuff_class_counter[int(cid)] += 1

        if (i + 1) % args.log_every == 0 or (i + 1) == n_probe:
            running_zero = zero_thing_samples / (i + 1) * 100.0
            running_mean_t = statistics.mean(thing_counts)
            running_mean_s = statistics.mean(stuff_counts)
            log.info(
                f"[{i+1:>4}/{n_probe}] zero_thing={running_zero:5.1f}%  "
                f"mean_thing={running_mean_t:4.2f}  mean_stuff={running_mean_s:4.2f}"
            )

    # Final summary.
    print("\n" + "=" * 72)
    print("PROBE SUMMARY")
    print("=" * 72)
    print(f"samples probed: {n_probe}")
    print(f"dataset length: {dataset_size}")
    print(summarise("thing_targets_per_crop", thing_counts))
    print(summarise("stuff_targets_per_crop", stuff_counts))
    zero_pct = zero_thing_samples / n_probe * 100.0
    print(
        f"crops_with_zero_things: {zero_thing_samples}/{n_probe} = {zero_pct:.1f}%"
    )
    print(
        f"crops_with_ge1_things: "
        f"{n_probe - zero_thing_samples}/{n_probe} = {100.0 - zero_pct:.1f}%"
    )

    print("\nthing class frequency (top 20, class_id in [0, num_thing)):")
    for cid, c in thing_class_counter.most_common(20):
        print(f"  class {cid:3d}: {c:6d} instances")

    print("\nstuff class frequency (top 20, class_id in dataset label space):")
    for cid, c in stuff_class_counter.most_common(20):
        print(f"  class {cid:3d}: {c:6d} occurrences")

    # Hypothesis D verdict.
    print("\n" + "=" * 72)
    if zero_pct >= 30.0:
        print(
            f"VERDICT: HYPOTHESIS D SUPPORTED ({zero_pct:.1f}% >= 30%). "
            "Too many crops have zero thing instances — thing queries rarely "
            "receive gradient. Consider crop strategy (bias toward non-empty), "
            "lower A_min in pseudo-label generation, or oversample thing-rich "
            "images."
        )
    elif zero_pct >= 15.0:
        print(
            f"VERDICT: HYPOTHESIS D PARTIALLY SUPPORTED ({zero_pct:.1f}% in "
            "[15%, 30%)). Crop emptiness contributes but is not the sole "
            "cause. Expect PQ_things gain from oversampling but also check "
            "training budget / query under-training."
        )
    else:
        print(
            f"VERDICT: HYPOTHESIS D RULED OUT ({zero_pct:.1f}% < 15%). "
            "Crops almost always contain things — PQ_things=0 is NOT caused "
            "by empty crops. Look at training budget, learning rate schedule, "
            "query under-training, or matcher cost balance."
        )
    print("=" * 72)


if __name__ == "__main__":
    main()
