import numpy as np
import torch
import torch.nn.functional as F

from mbps_pytorch.ga_uniap.pooling import ga_aggo_merge, _affinity


def _reference_vanilla(features, thresholds, min_size):
    """Verbatim cosine-only agglomerative merge (UniAP reference) for parity."""
    H, W, C = features.shape
    f = torch.from_numpy(features).reshape(H * W, C).float()
    fn = F.normalize(f, dim=1)
    clusters = [{"mask": (np.arange(H * W) == i), "nf": fn[i], "f": f[i],
                 "n": 1, "nb": set()} for i in range(H * W)]
    active = set(range(H * W))
    sims = {}
    for idx in range(H * W):
        if idx % W != 0:
            clusters[idx]["nb"].add(idx - 1); clusters[idx - 1]["nb"].add(idx)
            sims[(idx - 1, idx)] = float(fn[idx - 1] @ fn[idx])
        if idx - W >= 0:
            clusters[idx]["nb"].add(idx - W); clusters[idx - W]["nb"].add(idx)
            sims[(idx - W, idx)] = float(fn[idx - W] @ fn[idx])
    cur = H * W
    for th in thresholds:
        while sims:
            (i, j) = max(sims, key=sims.get)
            if sims[(i, j)] < th:
                break
            c1, c2 = clusters[i], clusters[j]
            ws = (c1["f"] + c2["f"]) / (c1["n"] + c2["n"])
            merged = {"mask": c1["mask"] | c2["mask"], "nf": F.normalize(ws, dim=0),
                      "f": c1["f"] + c2["f"], "n": c1["n"] + c2["n"],
                      "nb": (c1["nb"] | c2["nb"]) - {i, j}}
            clusters.append(merged); del sims[(i, j)]
            active.discard(i); active.discard(j); active.add(cur)
            for nb in merged["nb"]:
                for a in (i, j):
                    lo, hi = min(a, nb), max(a, nb)
                    if (lo, hi) in sims:
                        del sims[(lo, hi)]
                    clusters[nb]["nb"].discard(a)
                sims[(nb, cur)] = float(clusters[nb]["nf"] @ merged["nf"])
                clusters[nb]["nb"].add(cur)
            cur += 1
    out = [clusters[k]["mask"].reshape(H, W) for k in sorted(active)
           if clusters[k]["n"] >= min_size]
    return np.stack(out) if out else np.zeros((0, H, W), bool)


def test_reduces_to_vanilla_when_no_geometry():
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((8, 8, 16)).astype(np.float32)
    th, ms = (0.6, 0.4), 2
    got = ga_aggo_merge(feats, None, None, th, ms, w_f=1.0, w_n=0.0, w_h=0.0)
    ref = _reference_vanilla(feats, th, ms)
    assert got.shape == ref.shape
    gs = sorted([m.tobytes() for m in got]); rs = sorted([m.tobytes() for m in ref])
    assert gs == rs


def test_masks_are_disjoint_partition():
    rng = np.random.default_rng(1)
    feats = rng.standard_normal((8, 8, 16)).astype(np.float32)
    masks = ga_aggo_merge(feats, None, None, (0.6,), 1, 1.0, 0.0, 0.0)
    cover = masks.sum(axis=0)
    assert cover.max() <= 1, "clusters overlap"


def test_geometry_term_separates_equal_feature_regions():
    feats = np.ones((4, 8, 8), np.float32)
    normal = np.zeros((4, 8, 3), np.float32)
    normal[:, :4] = [0, 0, 1]; normal[:, 4:] = [1, 0, 0]
    height = np.zeros((4, 8), np.float32)
    geo = ga_aggo_merge(feats, normal, height, (0.6,), 1, w_f=0.5, w_n=0.5, w_h=0.0)
    none = ga_aggo_merge(feats, None, None, (0.6,), 1, w_f=1.0, w_n=0.0, w_h=0.0)
    assert geo.shape[0] > none.shape[0], "geometry failed to split equal-feature halves"


def test_affinity_reduces_to_cosine():
    # _affinity takes ALREADY-normalized features (ga_aggo_merge passes nf[a]).
    nf_a = F.normalize(torch.tensor([1.0, 0.0]), dim=0)
    nf_b = F.normalize(torch.tensor([0.5, 0.5]), dim=0)
    a = _affinity(nf_a, nf_b, None, None, None, None, 1.0, 0.0, 0.0)
    cos = float(nf_a @ nf_b)
    assert abs(a - cos) < 1e-6
