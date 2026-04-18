"""Stage-3 self-training orchestrator for M2 -> M2 fine-tuning.

Iterates SELF_TRAINING.ROUNDS rounds. In each round:
  1. Use the current model (loaded from CHECKPOINT) to predict pseudo
     panoptic labels on DATA.ROOT (train split).
  2. Filter by confidence threshold (N5 filter_by_confidence).
  3. Train for ROUND_STEPS with the new pseudo labels.
  4. Save the new checkpoint + EMA teacher as next round's source.
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess

from cups.config import get_default_config

logging.basicConfig(format="%(message)s", level=logging.INFO)
log = logging.getLogger(__name__)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    cfg = get_default_config(args.config)
    rounds = cfg.SELF_TRAINING.ROUNDS
    for r in range(rounds):
        tau = cfg.SELF_TRAINING.SEMANTIC_SEGMENTATION_THRESHOLD + r * cfg.SELF_TRAINING.CONFIDENCE_STEP
        log.info("=== Stage-3 Round %d/%d (tau=%.2f) ===", r + 1, rounds, tau)
        subprocess.run([
            "python", "refs/cups/generate_pseudo_labels.py",
            "--config", args.config,
            "--round", str(r),
            "--tau", str(tau),
            "--out", os.path.join(args.output, f"round_{r:02d}_labels"),
        ], check=True)
        subprocess.run([
            "python", "refs/cups/train_pseudo.py",
            "--config", args.config,
            "--round", str(r),
            "--pseudo_root", os.path.join(args.output, f"round_{r:02d}_labels"),
            "--output", os.path.join(args.output, f"round_{r:02d}_ckpt"),
        ], check=True)


if __name__ == "__main__":
    main()
