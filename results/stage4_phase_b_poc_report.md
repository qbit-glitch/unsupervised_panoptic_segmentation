# Phase-B POC: Cluster Reorg Diagnostic on existing k=80 centroids

**Date:** 2026-04-30
**Script:** `scripts/stage4_cluster_poc.py`
**Inputs:** `weights/kmeans_centroids_k80_santosh.npz` (80 centroids × 90 dim, with cluster_to_class mapping)
**Goal:** validate the rare-mode-freeze heuristic before paying for NeCo training compute.

---

## 1. Class coverage (19-class TrainID space)

- 80 clusters Hungarian-aligned to 17 of 19 Cityscapes classes
- **Vacant classes**: traffic_light (6), motorcycle (17)
- **Single-cluster classes** (analogs to rare modes): pole, traffic sign, rider, truck, train, bicycle

> Note: the local centroids use 19-class TrainID. The paper §4's "6 dead classes" (parking, guard rail, tunnel, polegroup, caravan, trailer) are from the 27-class CAUSE+Hungarian pipeline. To run the same diagnostic against the paper's dead classes, we need a 27-class re-aligned centroid file (likely on remote).

## 2. Pairwise centroid cosine similarity

| Statistic | Value |
|---|---|
| min | -0.356 |
| median | 0.315 |
| mean | 0.353 |
| max | 0.989 |
| pairs with cos > 0.95 | 47 |

47 obviously-redundant pairs out of 3160 total — these would be merged first by t-NEB.

## 3. Rare-mode candidates and their nearest neighbors

| Centroid | Class | Top neighbor cos | Top neighbor class | Verdict |
|---|---|---|---|---|
| 16 | train | 0.466 | bus | well-isolated |
| 21 | pole | 0.908 | sidewalk | heavily entangled |
| 32 | bicycle | 0.807 | rider | naturally adjacent |
| 45 | rider | 0.807 | bicycle | naturally adjacent |
| 46 | truck | 0.418 | vegetation | well-isolated |
| 78 | traffic sign | 0.828 | building | heavily entangled |

**Implication**: rare modes split into two regimes.
- *Well-isolated* (train, truck): freezing them prevents Hungarian collapse — α-3 alone should help.
- *Heavily entangled* (pole, traffic sign): freezing alone is insufficient because the centroid's geometry already overlaps with a frequent neighbor; **NeCo feature sharpening is essential** for these. If NeCo cannot separate them, α will not save these classes.
- *Naturally adjacent* (rider, bicycle): high similarity is expected and not a problem.

## 4. End-to-end merge sanity check

| | |
|---|---|
| Input | (80, 90) centroids, proxy population counts |
| Frozen | 6 single-cluster centroid IDs |
| Target | k=40 |
| Output | (40, 90) merged centroids, mapping (80,) |
| Frozen survived alone | True |
| Heuristic ↔ ground truth overlap | 6 / 6 |

Algorithm is correct and the rare-mode-freeze heuristic correctly identifies the same centroids as direct lookup of single-cluster classes.

## 5. Decision

> **GO with caution.** Median similarity 0.315 indicates moderate separation, sufficient for the algorithm to operate meaningfully. However, two single-cluster centroids (pole, traffic sign) are highly entangled with frequent-class neighbors (cos ~0.9). For these, hierarchical merge alone is **not enough** — NeCo feature sharpening or depth-conditioned splitting is required.

**Recommendation:** proceed with NeCo training, but add an explicit guard check after NeCo: re-run this diagnostic on the NeCo-sharpened centroids and confirm the median similarity drops AND the entangled rare modes (pole, traffic sign) move to lower max-neighbor similarity. If they do not, pivot to depth-conditioned splitting before paying for full Stage-2 retrain.

## 6. Limitations of this POC

- Local centroids are 19-class TrainID, not the 27-class paper space. The paper's 6 dead classes (parking, guard rail, tunnel, polegroup, caravan, trailer) are not directly testable here.
- Pixel counts are not available locally (the raw `pseudo_semantic_raw_k80/` is on remote). We used per-class cluster count as a proxy. Real pixel counts might shift the rare-mode candidates.
- The 90-dim CAUSE feature space is not the same as the DINOv3 ViT-B/16 768-dim space NeCo would operate on; the absolute similarity numbers are not directly comparable.
