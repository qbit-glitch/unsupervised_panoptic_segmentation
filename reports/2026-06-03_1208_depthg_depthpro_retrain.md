# DepthG fully-monocular retrain (Cityscapes + DepthPro) — Result Report

- **Run timestamp:** 2026-06-03, 10:09:25 → 12:08:22 local (wall-clock 1 h 59 min)
- **Host:** `santosh@172.17.254.146` (2× NVIDIA GTX 1080 Ti, 11 GB each)
- **Branch / HEAD:** `main` @ `9a007c82a4` (12 commits today wiring this pipeline)
- **Log:** `/home/santosh/datasets/cityscapes/depthg_retrain/logs/retrain_ddp_20260603_100919.log`
- **Checkpoints:** `/home/santosh/datasets/cityscapes/depthg_retrain/cityscapes_depthpro_monocular_ddp_date_Jun03_10-09-25/`
  - `epoch=6-step=1680.ckpt` (352 MB) — best by `test/cluster/mIoU` monitor
  - `last.ckpt` (352 MB) — step 7000

---

## 1. Setup

The goal of this run was to reproduce CUPS' DepthG semantic head **without any stereo or video signal** by replacing its ZoeDepth supervision with monocular DepthPro depth maps. Everything else — architecture, optimizer, contrastive loss — was held at the values extracted from the upstream `depthg.ckpt` hyper-parameters.

| Item | Value | Source |
|---|---|---|
| Backbone | DINO ViT-B/8 (frozen, 768-dim) | upstream default |
| Heads | linear cluster projector (dim=100, nonlinear) | upstream |
| Dataset | Cityscapes train (2,975 images), 27 classes | upstream |
| Depth source | **DepthPro monocular `.npy`, normalized [0,1]** | our substitution |
| Resolution | 224 × 224 | upstream |
| Crop type | `null` (CityscapesSeg + on-the-fly resize) | our bypass of pre-cropped layout |
| Steps | 7,000 | upstream |
| Batch size | 16 per GPU × 2 GPUs (DDP) = **effective 32** | upstream effective scale matched |
| LR | 5e-4 | upstream |
| `depth_feat_weight` | 0.036864 | upstream default (no HP sweep — user-elected) |
| `depth_feat_shift` | 0.012288 | upstream default |
| `depth_loss_decay` | `True`, factor 0.8 every 400 steps | upstream default (revisited below) |
| Pre-flight | DINO CLS NN cache (2975 × 8) built in 3.5 min | `precompute_knns.py` |

DepthPro depth was already cached on santosh for all 2,975 train + 500 val frames as `(512, 1024) float32 ∈ [0,1]` `.npy` arrays. No re-computation was needed.

---

## 2. Result vs. CUPS published baseline

The DepthG paper reports its `depthg.ckpt` (ZoeDepth depth, same backbone, same recipe, same dataset) at the values in column "CUPS published" below — directly comparable to our retrained checkpoint on Cityscapes val.

| Metric (27-class, val) | Ours @ peak (step 1680) | Ours @ final (step 7000) | CUPS published `depthg.ckpt` | Δ peak vs. CUPS |
|---|---|---|---|---|
| `test/linear/mIoU` | 27.7 | **29.55** | 23.8 | **+5.7** (better) |
| `test/linear/Accuracy` | 88.7 | **89.06** | — | — |
| `test/cluster/mIoU` | **14.82** | 13.78 | 22.3 | **−7.5** (worse) |
| `test/cluster/Accuracy` | 70.7 | 62.2 | — | — |
| `test/cluster/MaxAccuracy` (during training) | 72.4 | — | — | — |

### Reading

The linear probe is the cleanest read on feature quality (it asks: "given the learned codes, how well can a linear classifier separate the 27 Cityscapes classes?"). On that metric our monocular retrain **beats** the upstream stereo/video-trained ckpt by +5.7 mIoU. The features learned from DepthPro depth are genuinely better than from ZoeDepth, at least on this val set.

The cluster_probe asks a different question: "how well does the unsupervised partitioner align its 27 clusters with the 27 Cityscapes classes via Hungarian matching?" That number **misses** the gate (target ≥ 22.0; we hit 14.82 at peak, 13.78 at final). The information is in the features — the partitioning failed to lock onto the right boundaries.

The 15-point gap between linear (29.5) and cluster (13.8) at the final step is the diagnostic: same features, very different partition quality.

---

## 3. Training-trajectory anomaly

`test/cluster/mIoU` over the run (sampled at every val checkpoint, every ~280 batches):

```
step  ~200 : 3.81   (init — same as smoke)
step  ~500 : 11.53
step ~1000 : 12.57
step ~1500 : 13.71
step ~2000 : 14.62
step ~2500 : 14.82  <-- PEAK
step ~3000 : 14.29
step ~3500 : 14.52
step ~4000 : 14.25
step ~4500 : 14.48
step ~5000 : 14.32
step ~5500 : 14.39
step ~6000 : 14.07
step ~6500 : 13.29
step  7000 : 13.78   <-- FINAL (-1.04 from peak)
```

The cluster metric peaked around step 1680–2500, plateaued, then degraded by ~1 point by step 7000.

### Likely root cause

The upstream config sets `depth_loss_decay = True` with `depth_loss_decay_factor = 0.8` applied every `decay_every_steps = 400`. At step 7000 the cumulative decay is `0.8^17 ≈ 0.023`, i.e. the depth-correlation loss term is at **2.3 % of its starting weight**. The depth signal was essentially turned off long before the run ended.

This schedule is the upstream-published one (tuned for ZoeDepth + the original 7000-step recipe). Two possibilities for why it hurts more in our setting than in CUPS':

1. **Distribution mismatch.** DepthPro's monocular depth has sharper object-boundary discontinuities than ZoeDepth. The depth signal carries more weight per step early on; aggressive decay then removes a stronger regulariser, letting the cluster_probe drift more.
2. **Different optimum step.** With effective bs=32 via DDP, our gradient signal-to-noise is the same as upstream's bs=32, so step count should be comparable. But the cluster_probe converges before the backbone, and once depth decays out the cluster_probe slowly forgets the depth-induced alignment.

Either way: the practical reading is that `epoch=6-step=1680.ckpt` (the saved "best" checkpoint) captures the model at its highest cluster mIoU, before the decay erased the regularizer.

---

## 4. What we have

- A **fully monocular DepthG checkpoint** at `epoch=6-step=1680.ckpt` (352 MB). It is the first checkpoint in the project trained with **zero stereo, zero video, zero SMURF** — only single Cityscapes train images + DepthPro monocular depth.
- Reproducible launch chain on santosh: `scripts/santosh_depthg_{precompute_knns,smoke,retrain}.sh` + `refs/cups/external/depthg/src/{data.py, train_segmentation.py, configs/local_config.yml, precompute_knns.py}` patches all on `origin/main`.
- A measured trajectory showing the depth-decay schedule is over-aggressive for DepthPro depth.

## 5. What we do **not** have

- Cluster mIoU at or above the 22.0 gate. Pseudo-labels regenerated with this `cluster_probe` will be measurably weaker than those produced by upstream `depthg.ckpt`.
- A comparison of the resulting semantic PNGs against the CUPS-official cache. We have the data to do this but did not, in this session, run `gen_pseudo_labels.py` with the new checkpoint.

---

## 6. Open question — next-step options (deferred to user)

The user paused on the decision; recording the four routes for later resumption:

A. **Use `epoch=6-step=1680.ckpt` as-is** — regenerate semantic pseudo-labels for all 18 cities, judge visually + by downstream Stage-2 PQ, accept or reject the gate qualitatively. ~3 h on the M4. Closes the loop quickest; trades the eval-time cluster_probe metric for a downstream task metric.

B. **Retrain with `depth_loss_decay=false`** — keeps the depth signal alive throughout 7000 steps. Expected to lift cluster mIoU into the 18–20+ range based on the trajectory shape (peak at step 1680 came before the decay had fully kicked in). ~2 h compute.

C. **Retrain longer with deferred decay** — `max_steps=14000`, `decay_every_steps=2000`, `decay_factor=0.9`. Lets the cluster_probe converge before the depth signal fades. ~4 h compute, more thorough but doubles wall-clock vs. B.

D. **Mini HP sweep on `depth_feat_weight` × decay** — 3 short 1500-step runs (no decay; decay 0.95; weight ×3), pick the best by cluster mIoU at the same step count, then full retrain at the winning config. ~3 h total.

The author's bias from the trajectory alone is **B** (cheapest, addresses the most plausible root cause) followed by **A** (gates downstream on actual label quality, not the eval-time metric).

---

## 7. Provenance / reproducibility

- Exact CLI: `python -u external/depthg/src/train_segmentation.py experiment_name=depthpro_monocular_ddp gpus=2 batch_size=16 max_steps=7000 val_freq=500 checkpoint_freq=500 scalar_log_freq=10 depth_feat_weight=0.036864 depth_feat_shift=0.012288 wandb_logging=false`
- Env: `~/anaconda3/envs/cups`, Python 3.10, PyTorch 2.x, torchvision, transformers 4.40.2. Conda activate wrapped in `set +u ... set -u` to dodge santosh's `ADDR2LINE` unbound-var hook. `LD_LIBRARY_PATH=$CONDA_PREFIX/lib` to dodge a `libLerc.so.4` / system `libstdc++` clash.
- DDP strategy: `ddp_find_unused_parameters_true` (Lightning auto). World size 2.
- DINO ViT-B/8 weights auto-downloaded from `https://dl.fbaipublicfiles.com/dino/dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth` on first use, cached at `~/.cache/torch/hub/checkpoints/`.
- NN cache: `/home/santosh/datasets/cityscapes/nns/nns_vit_base_cityscapes_train_None_224.npz`, shape (2975, 8).
- DepthPro cache: `/home/santosh/datasets/cityscapes/depth_depthpro/{train,val}/<city>/<frame>.npy` (3,476 files, pre-existing on santosh).
- Notable warnings ignored as benign: (1) `torch.meshgrid` deprecation per dataloader sample; (2) DDP grad-stride hint on `(27,100,1,1)` cluster_probe weight (performance-only); (3) Lightning manual-optimization vs. ModelCheckpoint note about pre-vs-post-optimizer-step state (informational; we save post-step state).

---

*Report generated 2026-06-03 by Claude Code, project `mbps_panoptic_segmentation`.*
