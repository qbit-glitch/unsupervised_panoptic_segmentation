import sys
from typing import List, Tuple

from mbps_pytorch.ga_uniap.config import Phase0Config


def list_val_stems(cfg: Phase0Config) -> List[Tuple[str, str]]:
    """Return (stem, city) pairs that have BOTH DepthPro depth and gtFine."""
    root = cfg.data_root
    if not root.exists():
        raise FileNotFoundError(
            f"Cityscapes drive not mounted at {root}. Mount /Volumes/code_files first."
        )
    depth_root = root / "depth_depthpro" / cfg.split
    gt_root = root / "gtFine" / cfg.split
    out: List[Tuple[str, str]] = []
    for city_dir in sorted(depth_root.glob("*")):
        if not city_dir.is_dir():
            continue
        city = city_dir.name
        for npy in sorted(city_dir.glob("*.npy")):
            stem = npy.stem
            gt = gt_root / city / f"{stem}_gtFine_labelTrainIds.png"
            inst = gt_root / city / f"{stem}_gtFine_instanceIds.png"
            if gt.exists() and inst.exists():
                out.append((stem, city))
            if len(out) >= cfg.n_images:
                return out
    return out


if __name__ == "__main__":
    cfg = Phase0Config()
    stems = list_val_stems(cfg)
    print(f"usable val stems: {len(stems)} (requested {cfg.n_images})")
    for s, c in stems[:3]:
        print(" ", c, s)
    sys.exit(0 if stems else 1)
