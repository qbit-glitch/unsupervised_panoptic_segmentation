"""Geometric agglomerative pooling — adapted from S2-UniSeg FastUniAP.aggo_merge.

Single edge affinity blends appearance cosine with depth-derived geometry:
    S_ij = w_f*cos(f_i,f_j) + w_n*(n_i . n_j) + w_h*(1 - |h_i-h_j|/H_SCALE)
With w_n=w_h=0 it is exactly the vanilla UniAP cosine merge.
"""
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

H_SCALE = 3.0  # metres; height-difference scale for the height-similarity term


def _affinity(nf_i, nf_j, n_i, n_j, h_i, h_j, w_f, w_n, w_h) -> float:
    s = w_f * float(nf_i @ nf_j)
    if w_n and n_i is not None:
        s += w_n * float(n_i @ n_j)
    if w_h and h_i is not None:
        hsim = 1.0 - abs(float(h_i) - float(h_j)) / H_SCALE
        s += w_h * max(-1.0, min(1.0, hsim))
    return s


def ga_aggo_merge(features: np.ndarray,
                  normal: Optional[np.ndarray],
                  height: Optional[np.ndarray],
                  thresholds: Tuple[float, ...],
                  min_size: int,
                  w_f: float, w_n: float, w_h: float) -> np.ndarray:
    """Return (K, gh, gw) bool disjoint cluster masks (segments >= min_size)."""
    H, W, C = features.shape
    f = torch.from_numpy(features).reshape(H * W, C).float()
    nf = F.normalize(f, dim=1)
    nrm = None if normal is None else torch.from_numpy(normal).reshape(H * W, 3).float()
    hgt = None if height is None else torch.from_numpy(height).reshape(H * W).float()

    def aff(a, b):
        return _affinity(nf[a], nf[b], None if nrm is None else nrm[a],
                         None if nrm is None else nrm[b],
                         None if hgt is None else hgt[a],
                         None if hgt is None else hgt[b], w_f, w_n, w_h)

    clusters = [{"mask": (np.arange(H * W) == i), "nf": nf[i], "f": f[i],
                 "nrm": None if nrm is None else nrm[i],
                 "h": None if hgt is None else float(hgt[i]),
                 "n": 1, "nb": set()} for i in range(H * W)]
    sims = {}
    for idx in range(H * W):
        if idx % W != 0:
            clusters[idx]["nb"].add(idx - 1); clusters[idx - 1]["nb"].add(idx)
            sims[(idx - 1, idx)] = aff(idx - 1, idx)
        if idx - W >= 0:
            clusters[idx]["nb"].add(idx - W); clusters[idx - W]["nb"].add(idx)
            sims[(idx - W, idx)] = aff(idx - W, idx)

    def caff(a, b):
        ca, cb = clusters[a], clusters[b]
        s = w_f * float(ca["nf"] @ cb["nf"])
        if w_n and ca["nrm"] is not None:
            s += w_n * float(F.normalize(ca["nrm"], dim=0) @ F.normalize(cb["nrm"], dim=0))
        if w_h and ca["h"] is not None:
            hsim = 1.0 - abs(ca["h"] - cb["h"]) / H_SCALE
            s += w_h * max(-1.0, min(1.0, hsim))
        return s

    cur = H * W
    for th in thresholds:
        while sims:
            (i, j) = max(sims, key=sims.get)
            if sims[(i, j)] < th:
                break
            c1, c2 = clusters[i], clusters[j]
            tot = c1["n"] + c2["n"]
            ws = (c1["f"] + c2["f"]) / tot
            merged = {
                "mask": c1["mask"] | c2["mask"], "nf": F.normalize(ws, dim=0),
                "f": c1["f"] + c2["f"], "n": tot,
                "nrm": None if c1["nrm"] is None else (c1["nrm"] * c1["n"] + c2["nrm"] * c2["n"]) / tot,
                "h": None if c1["h"] is None else (c1["h"] * c1["n"] + c2["h"] * c2["n"]) / tot,
                "nb": (c1["nb"] | c2["nb"]) - {i, j},
            }
            clusters.append(merged); del sims[(i, j)]
            for nb in merged["nb"]:
                for a in (i, j):
                    lo, hi = min(a, nb), max(a, nb)
                    if (lo, hi) in sims:
                        del sims[(lo, hi)]
                    clusters[nb]["nb"].discard(a)
                sims[(nb, cur)] = caff(nb, cur)
                clusters[nb]["nb"].add(cur)
            cur += 1

    seen, out = set(), []
    for (m, n) in sims:
        for k in (m, n):
            if k not in seen:
                seen.add(k)
                if clusters[k]["n"] >= min_size:
                    out.append(clusters[k]["mask"].reshape(H, W))
    return np.stack(out) if out else np.zeros((0, H, W), bool)
