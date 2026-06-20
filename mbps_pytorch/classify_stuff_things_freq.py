"""GT-free thing/stuff split via CUPS frequency-ratio (psi=0.08).

A cluster is a "thing" if it co-occurs with >1 instance in a large enough
fraction of the images it appears in. Uses only pseudo-labels — no GT.
"""
import argparse
import glob
import json
import os
from typing import List, Set, Tuple

import numpy as np
from PIL import Image


def classify_from_arrays(sems: List[np.ndarray], insts: List[np.ndarray],
                         num_clusters: int = 27, threshold: float = 0.08
                         ) -> Tuple[Set[int], Set[int]]:
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


def classify_stuff_things(semantic_dir: str, instance_dir: str,
                          num_clusters: int = 27, threshold: float = 0.08
                          ) -> Tuple[Set[int], Set[int]]:
    sems, insts = [], []
    for sp in sorted(glob.glob(os.path.join(semantic_dir, "*", "*.png"))):
        stem = os.path.basename(sp).replace("_leftImg8bit.png", "").replace(".png", "")
        city = os.path.basename(os.path.dirname(sp))
        cand = [os.path.join(instance_dir, city, f"{stem}.png"),
                os.path.join(instance_dir, city, f"{stem}_leftImg8bit.png")]
        ip = next((p for p in cand if os.path.exists(p)), None)
        if ip is None:
            continue
        sems.append(np.array(Image.open(sp), dtype=np.int32))
        insts.append(np.array(Image.open(ip), dtype=np.int32))
    return classify_from_arrays(sems, insts, num_clusters, threshold)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--semantic_dir", required=True)
    p.add_argument("--instance_dir", required=True)
    p.add_argument("--num_clusters", type=int, default=27)
    p.add_argument("--threshold", type=float, default=0.08)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    things, stuff = classify_stuff_things(a.semantic_dir, a.instance_dir,
                                          a.num_clusters, a.threshold)
    json.dump({"thing_clusters": sorted(things), "stuff_clusters": sorted(stuff)},
              open(a.output, "w"), indent=2)
    print(f"things={sorted(things)}\nstuff={sorted(stuff)}\n→ {a.output}")


if __name__ == "__main__":
    main()
