"""Sanity-gate verifier for E1 Stage-1 pseudo-label regeneration.

Catches the 84.5%-empty-instances failure mode that hit our 1080 Ti run by
checking, on disk, that:

    (a) the expected number of (semantic, instance) pairs were written,
    (b) the fraction of all-zero instance maps stays below a threshold,
    (c) semantic maps cover the expected number of classes (~27 for CUPS),
    (d) per-image semantic / instance dimensions match.

Exits non-zero if any gate fails so the launcher script aborts before Stage-2.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO)
logger = logging.getLogger("e1_verify")


@dataclass
class VerifyReport:
    pseudo_root: str
    expected_count: int
    actual_pairs: int
    semantic_only: int
    instance_only: int
    empty_instance_count: int
    empty_instance_frac: float
    semantic_class_counts: dict
    sample_resolutions: List[List[int]]
    passed: bool
    failures: List[str]


def _load_png_as_array(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p))


def verify(pseudo_root: Path, expected_count: int, max_empty_frac: float,
           min_unique_classes: int, sample_size: int) -> VerifyReport:
    sem_files = sorted(pseudo_root.glob("*_semantic.png"))
    inst_files = sorted(pseudo_root.glob("*_instance.png"))
    sem_stems = {p.name.replace("_semantic.png", "") for p in sem_files}
    inst_stems = {p.name.replace("_instance.png", "") for p in inst_files}
    common = sem_stems & inst_stems

    failures: List[str] = []
    if expected_count > 0 and len(common) < expected_count:
        failures.append(
            f"only {len(common)} matched pairs, expected {expected_count}"
        )

    semantic_only = len(sem_stems - inst_stems)
    instance_only = len(inst_stems - sem_stems)

    empty = 0
    sem_class_counter: Counter = Counter()
    sample_res: List[List[int]] = []
    sample_stems = sorted(common)[: sample_size if sample_size > 0 else len(common)]
    for stem in sample_stems:
        inst = _load_png_as_array(pseudo_root / f"{stem}_instance.png")
        sem = _load_png_as_array(pseudo_root / f"{stem}_semantic.png")
        if int(inst.max()) == 0:
            empty += 1
        sem_class_counter.update(np.unique(sem).tolist())
        if len(sample_res) < 5:
            sample_res.append(list(sem.shape))

    sample_n = max(1, len(sample_stems))
    empty_frac = empty / sample_n
    if empty_frac > max_empty_frac:
        failures.append(
            f"empty-instance fraction {empty_frac:.3f} exceeds {max_empty_frac:.3f} "
            f"(was 0.845 on the broken 1080 Ti run)"
        )
    if len(sem_class_counter) < min_unique_classes:
        failures.append(
            f"only {len(sem_class_counter)} unique semantic ids across sample "
            f"(expected >= {min_unique_classes}; CUPS uses 27 clusters)"
        )

    report = VerifyReport(
        pseudo_root=str(pseudo_root),
        expected_count=expected_count,
        actual_pairs=len(common),
        semantic_only=semantic_only,
        instance_only=instance_only,
        empty_instance_count=empty,
        empty_instance_frac=empty_frac,
        semantic_class_counts={int(k): int(v) for k, v in sem_class_counter.items()},
        sample_resolutions=sample_res,
        passed=not failures,
        failures=failures,
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pseudo_root", type=Path, required=True)
    parser.add_argument("--expected_count", type=int, default=2975,
                        help="Cityscapes train: 2975. Use 0 to skip the count gate.")
    parser.add_argument("--max_empty_frac", type=float, default=0.01,
                        help="Max allowed fraction of all-zero instance maps.")
    parser.add_argument("--min_unique_classes", type=int, default=20,
                        help="DepthG cluster_probe has 27 classes; require at least this many.")
    parser.add_argument("--sample_size", type=int, default=0,
                        help="If >0, only inspect this many pairs (else all).")
    parser.add_argument("--report", type=Path, default=None,
                        help="Optional JSON report path.")
    args = parser.parse_args()

    if not args.pseudo_root.is_dir():
        logger.error("pseudo_root does not exist: %s", args.pseudo_root)
        return 2

    report = verify(
        pseudo_root=args.pseudo_root,
        expected_count=args.expected_count,
        max_empty_frac=args.max_empty_frac,
        min_unique_classes=args.min_unique_classes,
        sample_size=args.sample_size,
    )

    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(asdict(report), indent=2))

    logger.info("pairs=%d (expected %d)  empty_inst=%.3f  unique_sem_ids=%d",
                report.actual_pairs, report.expected_count,
                report.empty_instance_frac, len(report.semantic_class_counts))
    if report.passed:
        logger.info("VERIFY PASSED")
        return 0
    for f in report.failures:
        logger.error("FAIL: %s", f)
    return 1


if __name__ == "__main__":
    sys.exit(main())
