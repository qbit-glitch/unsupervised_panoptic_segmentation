"""GT-free thing/stuff split via CUPS frequency-ratio (psi=0.08).

A cluster is a "thing" if it tends to appear as multiple objects. Two GT-free
modes:
  - instance mode: count distinct pseudo-instance IDs overlapping the cluster
    (needs an instance dir, e.g. pseudo_instance_depthpro).
  - cc mode (default when no instance dir): count connected components of the
    cluster's own semantic mask (needs only the k27 labels).
Both use only pseudo-labels — no GT.
"""
import argparse
import glob
import json
import os
from typing import List, Optional, Set, Tuple

import numpy as np
from PIL import Image


def classify_from_arrays(sems: List[np.ndarray], insts: List[np.ndarray],
                         num_clusters: int = 27, threshold: float = 0.08
                         ) -> Tuple[Set[int], Set[int]]:
    """Instance mode: thing if >1 distinct instance id overlaps the cluster."""
    appear = np.zeros(num_clusters, dtype=np.int64)
    multi = np.zeros(num_clusters, dtype=np.int64)
    for sem, inst in zip(sems, insts):
        for c in range(num_clusters):
            cm = sem == c
            if not cm.any():
                continue
            appear[c] += 1
            ids = np.unique(inst[cm])
            ids = ids[ids > 0]
            if len(ids) > 1:
                multi[c] += 1
    ratio = multi / (appear + 1e-9)
    things = set(int(c) for c in np.where(ratio > threshold)[0])
    return things, set(range(num_clusters)) - things


def classify_from_semantic_cc(sems: List[np.ndarray], num_clusters: int = 27,
                              threshold: float = 0.08, min_area: int = 64
                              ) -> Tuple[Set[int], Set[int]]:
    """Instance-free: thing if the cluster mask splits into >1 connected
    component (area>=min_area) in a >threshold fraction of the images it
    appears in. Needs only the semantic k27 labels."""
    from scipy import ndimage
    appear = np.zeros(num_clusters, dtype=np.int64)
    multi = np.zeros(num_clusters, dtype=np.int64)
    for sem in sems:
        for c in range(num_clusters):
            cm = sem == c
            if not cm.any():
                continue
            appear[c] += 1
            lab, n = ndimage.label(cm)
            if n <= 1:
                continue
            sizes = np.bincount(lab.ravel())[1:]  # drop background count
            if int((sizes >= min_area).sum()) > 1:
                multi[c] += 1
    ratio = multi / (appear + 1e-9)
    things = set(int(c) for c in np.where(ratio > threshold)[0])
    return things, set(range(num_clusters)) - things


def _load_sems_insts(semantic_dir: str, instance_dir: Optional[str]):
    sems, insts = [], []
    for sp in sorted(glob.glob(os.path.join(semantic_dir, "*", "*.png"))):
        sem = np.array(Image.open(sp), dtype=np.int32)
        if instance_dir is None:
            sems.append(sem)
            continue
        stem = os.path.basename(sp).replace("_leftImg8bit.png", "").replace(".png", "")
        city = os.path.basename(os.path.dirname(sp))
        cand = [os.path.join(instance_dir, city, f"{stem}.png"),
                os.path.join(instance_dir, city, f"{stem}_leftImg8bit.png")]
        ip = next((p for p in cand if os.path.exists(p)), None)
        if ip is None:
            continue
        sems.append(sem)
        insts.append(np.array(Image.open(ip), dtype=np.int32))
    return sems, insts


def classify_stuff_things(semantic_dir: str, instance_dir: Optional[str] = None,
                          num_clusters: int = 27, threshold: float = 0.08,
                          min_area: int = 64) -> Tuple[Set[int], Set[int]]:
    sems, insts = _load_sems_insts(semantic_dir, instance_dir)
    if instance_dir is not None:
        return classify_from_arrays(sems, insts, num_clusters, threshold)
    return classify_from_semantic_cc(sems, num_clusters, threshold, min_area)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_dir", required=True)
    p.add_argument("--instance_dir", default=None,
                   help="Optional pseudo-instance dir. If omitted, uses connected-"
                        "components of the semantic mask (instance-free).")
    p.add_argument("--num_clusters", type=int, default=27)
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--min_area", type=int, default=64)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    things, stuff = classify_stuff_things(a.semantic_dir, a.instance_dir,
                                          a.num_clusters, a.threshold, a.min_area)
    json.dump({"thing_clusters": sorted(things), "stuff_clusters": sorted(stuff)},
              open(a.output, "w"), indent=2)
    mode = "instance" if a.instance_dir else "cc"
    print(f"[{mode}] things={sorted(things)}\nstuff={sorted(stuff)}\n→ {a.output}")


if __name__ == "__main__":
    main()
