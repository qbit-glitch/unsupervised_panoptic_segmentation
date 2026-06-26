from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

DATA_ROOT = Path("/Volumes/code_files/datasets/cityscapes")


@dataclass(frozen=True)
class Phase0Config:
    data_root: Path = DATA_ROOT
    split: str = "val"
    grid_h: int = 32
    grid_w: int = 64
    work_h: int = 512
    work_w: int = 1024
    # Calibrated for DINOv3 ViT-B/16 (adjacent-patch cosine median ~0.955);
    # UniAP's original (0.8..0.4) fully collapses the graph on these features.
    thresholds: Tuple[float, ...] = (0.96, 0.94, 0.92, 0.90)
    min_size: int = 4
    n_images: int = 120
    device: str = "cpu"


# Affinity variants for the Phase-0 ablation (folded in from Task 5).
VARIANTS: Dict[str, dict] = {
    "V0_vanilla":   {"mode": "single", "weights": (1.0, 0.0, 0.0), "weights_things": None},
    "V1_augment":   {"mode": "single", "weights": (1.0, 0.5, 0.3), "weights_things": None},
    "V2_split":     {"mode": "split",  "weights": (1.0, 0.0, 0.0), "weights_things": (0.3, 0.6, 0.4)},
    "V3_geom_only": {"mode": "single", "weights": (0.0, 0.6, 0.4), "weights_things": None},
}


def weight_sweep() -> List[Tuple[float, float, float]]:
    """V1 augment tuning grid: w_f=1, w_n,w_h in {0.3,0.5,0.7}."""
    return [(1.0, wn, wh) for wn, wh in product((0.3, 0.5, 0.7), (0.3, 0.5, 0.7))]
