# Depth-Aware Superpixel-Guided Mask (SGM) Adapter — Design Reference

> **Scope.** This is a *design spec only*. It documents a proposed new branch in the
> Stage-1 instance pipeline that pairs the Hoang 2025 *Unsupervised Instance Segmentation
> with Superpixels* losses with a DINOv2 + DepthPro fused-feature adapter.
>
> **Do not modify** the existing superpixel-affinity adapter codebase. This adapter is a
> **sibling** to it, not a replacement.

---

## 1 Relationship to existing code

The repository already contains an edge-MLP merge-probability adapter; this spec adds a
pixel-level foreground-probability adapter trained under different losses.

| Concern | Existing edge-MLP adapter | This spec (SGM adapter) |
|---|---|---|
| Output | $p_\text{merge}(u, v) \in [0,1]$ per adjacent SP pair | $\widetilde M_c(p) \in [0,1]$ per pixel, per thing class |
| Training loss | weighted BCE on edge merge/split labels | $\mathcal{L}_\text{hard} + \mathcal{L}_\text{soft} + \lambda_\text{ad}\mathcal{L}_\text{ad}$ from Hoang 2025 |
| Decoding | threshold $p_\text{merge}$, CC per class | argmax over class foreground maps |
| Reference paper | internal ablation | `test-instance-labels/papers/Superpixels.pdf` (Hoang 2025) |
| Reuses superpixels? | yes (SLIC) | yes (depth-aware SLIC; see §3) |

**Existing files to read but not modify:**

- `mbps_pytorch/instance_methods/superpixel_affinity.py` — SLIC + per-SP descriptors, graph utilities
- `mbps_pytorch/models/instance/superpixel_affinity_adapter.py` — edge MLP
- `mbps_pytorch/train_superpixel_affinity_adapter.py` — trainer
- `mbps_pytorch/generate_superpixel_affinity_instances.py` — inference
- `mbps_pytorch/generate_depth_guided_instances.py::depth_guided_instances` — current $(\tau_d, A_\min, n_\text{dil})$ baseline that supplies the *coarse mask* used as supervision
- `mbps_pytorch/models/semantic/depth_adapter.py::sinusoidal_depth_encode` — reused for depth encoding
- `test-instance-labels/Superpixels/superpixels_uis/superpixels.py` — paper reference, do not modify
- `test-instance-labels/Superpixels/superpixels_uis/multicut.py` — paper reference, do not modify

---

## 2 Pipeline overview

```
Image x  ──► DINOv2 ViT-B/14  ─► f  (B, N, 768)
        │
        ├─► DepthPro          ─► d  (B, H', W')          → sinusoidal e(d)  (B, N, 16)
        │
        ├─► depth-aware SLIC  ─► S = {S_k}  (≈ 1500 SP / image)  [§3]
        │
        └─► current Algo. 4   ─► coarse mask M  (per thing class)
                                                      │
   Fuse(f, e(d))  ─►  ̃f  (B, N, D)  ──► SGM head ──► widetilde{M}_c (B, C_thing, H, W)
                                                       │
                                              Eq. 4 P_k(widetilde{M})
                                          ┌────────────┴────────────┐
                                          ▼                         ▼
                                  L_hard  (Eq. 5)         L_soft  (Eq. 6–8, depth-aware w_{m,n})
                                          └────────────┬────────────┘
                                                       ▼
                                        +  λ_ad · L_ad  (Eq. 10, holistic stability)
```

`C_thing = 8` (Cityscapes thing trainIDs 11–18). Resolutions are inherited from the
existing pipeline ($H' = 512, W' = 1024$, patch grid $32 \times 64$).

---

## 3 Depth-aware superpixels

Replace the SLIC stack used in the existing adapter (RGB-only or RGB+depth) with

$$
F_\text{sp}(p) \;=\; \big[\,\text{RGB}(p);\; \alpha_\text{slic}\,d(p);\; \beta_\text{slic}\,\mathrm{PCA}_8(f^{DINO}(p))\,\big] \in \mathbb{R}^{12},
$$

then call `scikit-image` SLIC on $F_\text{sp}$. Defaults: $\alpha_\text{slic} = 1.0$,
$\beta_\text{slic} = 0.5$, target compactness $10$, $n_\text{segments} \approx 1500$.

Cache per image as `cityscapes/superpixels_dinodepth_slic/{split}/{city}/{stem}.npy`
(int32 superpixel ID map). Generated once; not retrained.

---

## 4 Fusion module

Two architectures, both acceptable:

### 4A — Cross-attention transformer (preferred)

```python
class FusionXAttn(nn.Module):
    def __init__(self, d_dino=768, d_depth=16, d_out=768, n_layers=3, n_heads=8):
        ...
    def forward(self, f_dino, depth_enc):
        # f_dino:    (B, N, 768)
        # depth_enc: (B, N, 16)
        kv = self.proj_kv(depth_enc)            # (B, N, 768)
        x  = f_dino
        for layer in self.layers:
            x = x + layer.attn(q=x, k=kv, v=kv) # depth-conditioned attention
            x = x + layer.ffn(x)
        return x                                 # (B, N, 768)
```

~1.0 M params for 3 layers / 8 heads / 768 dim.

### 4B — Concat + MLP (cheap fallback)

$\tilde f = \mathrm{MLP}([f \,\Vert\, e(d)])$, two layers, hidden 384, ≈ 50 K params.
Use when latency / memory matter (1080 Ti).

---

## 5 SGM head

Project $\tilde f$ back to image resolution and emit per-class foreground logits.

```python
class SGMHead(nn.Module):
    def __init__(self, d_in=768, n_thing=8):
        self.up   = nn.ConvTranspose2d(d_in, 128, 4, 2, 1)   # 32x64 -> 64x128
        self.mid  = nn.Sequential(Conv3x3(128, 64), GELU(), Conv3x3(64, 32))
        self.head = nn.Conv2d(32, n_thing, kernel_size=1)
    def forward(self, fbar):  # (B, 768, 32, 64)  reshaped from (B, N, 768)
        x = self.up(fbar)
        x = F.interpolate(x, size=(H_prime, W_prime), mode="bilinear")
        return torch.sigmoid(self.head(self.mid(x)))   # (B, 8, 512, 1024)
```

---

## 6 Losses — depth-aware version of Hoang 2025 $\mathcal{L}_\text{sgm}$

For every superpixel $S_k$ compute three means once per image:

$$
\mu^c_k = \tfrac{1}{|S_k|}\!\!\sum_{i \in S_k}\! C_i, \quad
\bar d_k = \tfrac{1}{|S_k|}\!\!\sum_{i \in S_k}\! d_i, \quad
\bar f_k = \tfrac{1}{|S_k|}\!\!\sum_{i \in S_k}\! f^{DINO}_i.
$$

### 6.1 Pixel→superpixel weight (generalises Eq. 3)

$$
\delta_{k,i} \;=\; \exp\!\Big(-\tfrac{\|\mu^c_k - C_i\|_2^2}{\alpha_1}
                          \;-\; \tfrac{(\bar d_k - d_i)^2}{\alpha_d}\Big).
$$

Defaults: $\alpha_1 = 0.05$, $\alpha_d = 0.10$.

### 6.2 Superpixel foreground probability (Eq. 4, unchanged shape)

$$
P^{(c)}_k \;=\; \tfrac{1}{\nu_k}\sum_{i \in S_k} \widetilde M_c(i)\, \delta_{k,i},
\qquad \nu_k = \sum_{i \in S_k} \delta_{k,i}.
$$

### 6.3 Hard loss (Eq. 5, per class)

Coarse mask supervision $M_c$ from `depth_guided_instances`:

- $y^{(c)}_k = 1$ if all pixels in $S_k$ are foreground in $M_c$
- $y^{(c)}_k = 0$ if all are background
- otherwise $S_k$ is **unlabeled** for class $c$ (skip)

$$
\mathcal{L}_\text{hard} = -\tfrac{1}{C_\text{thing}\,N_s}\sum_c \sum_{k:\,\text{labelled}}
   \big( y^{(c)}_k \log P^{(c)}_k + (1 - y^{(c)}_k) \log(1 - P^{(c)}_k) \big).
$$

### 6.4 Soft loss (Eq. 6–8 with depth-aware MST edge weights)

Build the superpixel adjacency graph $\mathcal{G}$ with edge weights

$$
\boxed{\;w_{m,n} = \|\mu^c_m - \mu^c_n\|_2^2 \;+\; \lambda_d\,(\bar d_m - \bar d_n)^2 \;+\; \lambda_f\,\|\bar f_m - \bar f_n\|_2^2\;}
$$

Defaults: $\lambda_d = 5.0$, $\lambda_f = 1.0$.

Compute minimum-spanning tree $\mathcal{G}_T$, then path-max affinity

$$
\psi_{k,l} \;=\; \exp\!\Big(-\tfrac{1}{\alpha_2}\!\!\!\max_{(m,n) \in \mathbb{E}_{k,l}}\!\! w_{m,n}\Big),
\quad
\hat P^{(c)}_k \;=\; \tfrac{1}{\gamma_k}\sum_{l \in S} P^{(c)}_l\, \psi_{k,l}, \quad \gamma_k = \sum_l \psi_{k,l},
$$

with $\alpha_2 = 0.3$. The soft loss (Eq. 8) is

$$
\mathcal{L}_\text{soft} \;=\; \tfrac{1}{C_\text{thing}\,|S|} \sum_c \sum_k \big| P^{(c)}_k - \hat P^{(c)}_k \big|.
$$

### 6.5 Adaptive self-training (Eq. 10, unchanged)

Save $e = 4$ checkpoints during training. For each predicted mask $\mathbf m^e_i$ from
the last checkpoint, compute

$$
Z_i \;=\; \sum_{j=1}^{e-1} \mathrm{IoU}\big(\mathbf m^e_i,\, \mathbf m^j_i\big),
$$

min-max normalise $Z$ across all predicted masks in the dataset, and use the result as
per-mask weights in a final round of supervised refinement against $\widetilde M$
from the last checkpoint.

### 6.6 Total

$$
\mathcal{L} \;=\; \mathcal{L}_\text{hard} \;+\; \mathcal{L}_\text{soft}
              \;+\; \lambda_\text{ad}\,\mathcal{L}_\text{ad},
\qquad \lambda_\text{ad} = 1.0 \text{ (after warm-up)}.
$$

---

## 7 Inference and instance extraction

```
1.  For each image x:
2.      ̃f      ← Fusion(DINOv2(x), e(DepthPro(x)))
3.      M̃     ← SGMHead( ̃f )                            # (8, 512, 1024)
4.      For each thing class c ∈ {11..18}:
5.          fg_c ← (M̃[c] > 0.5)
6.          (L, n) ← ConnectedComponents( fg_c )         # 8-neighbour
7.          drop components with area < A_min            # reuse current A_min
8.          append non-empty masks to instance list with score = mean(M̃[c] inside mask)
9.      Save NPZ in the existing format expected by
        `mbps_pytorch/evaluate_cascade_pseudolabels.py`
```

The output schema matches `generate_depth_guided_instances.py::save_instances`:
`masks: (N, H'·W') bool`, `scores: (N,) float32`, `num_valid: int`, `h_patches`, `w_patches`.

---

## 8 New files needed (when this design is greenlit)

Only four new files; nothing existing is touched:

| File | Purpose |
|---|---|
| `mbps_pytorch/instance_methods/depth_aware_slic.py` | depth-aware SLIC superpixel generator |
| `mbps_pytorch/models/instance/sgm_adapter.py` | Fusion module + SGM head (Algorithms in §4–5) |
| `mbps_pytorch/losses/superpixel_sgm.py` | `L_hard`, `L_soft`, `L_ad` with depth-aware $w_{m,n}$ |
| `mbps_pytorch/train_sgm_adapter.py` | trainer wiring it all together; reuses existing `coarse mask` from `generate_depth_guided_instances` |

Inference reuses `generate_depth_guided_instances.py::save_instances` for the NPZ layout,
so `evaluate_cascade_pseudolabels.py` works unchanged.

---

## 9 Hyper-parameter cheat-sheet

| Symbol | Default | Notes |
|---|---:|---|
| $\alpha_1$ (color BW in $\delta$) | 0.05 | Eq. 3 |
| $\alpha_d$ (depth BW in $\delta$) | 0.10 | Eq. 3 |
| $\lambda_d$ (depth weight in $w$) | 5.0 | Eq. 6 |
| $\lambda_f$ (DINO weight in $w$)  | 1.0 | Eq. 6 |
| $\alpha_2$ (MST temperature)      | 0.3 | Eq. 6 |
| $\lambda_\text{ad}$ (adaptive)    | 1.0 | after warm-up |
| Superpixels per image $|S|$       | $\approx 1500$ | SLIC `n_segments` |
| Fusion module params              | $\approx 1$ M | 3-layer cross-attn |
| Head params                       | $\approx 0.1$ M | small conv |
| Total trainable                   | $\approx 1.1$ M | |
| Coarse mask source $M$            | `depth_guided_instances(S, D; τ_d, A_min, n_dil)` | unchanged |

---

## 10 Status

- [x] Design spec captured (this file)
- [x] Depth-aware SLIC implementation — `mbps_pytorch/instance_methods/depth_aware_slic.py`
- [x] Fusion + SGM head implementation — `mbps_pytorch/models/instance/sgm_adapter.py`
- [x] Loss module — `mbps_pytorch/losses/superpixel_sgm.py`
- [x] Trainer and CLI — `mbps_pytorch/train_sgm_adapter.py`
- [x] Smoke test (9/9 pass) — `mbps_pytorch/tests/test_sgm_adapter_smoke.py`
- [ ] Full Cityscapes train + downstream Stage-2 evaluation

## 11 Compute placement

| Component | Where it runs | Why |
|---|---|---|
| Depth-aware SLIC (`compute_or_load_slic`) | Local (MPS/CPU) | Pure scikit-image; cached per-image once. |
| SGM adapter forward + losses (`train_sgm_adapter.py`) | Local (MPS/CPU/CUDA) | ~1 M params; SciPy MST step is fast on CPU. |
| RAMA MultiCut coarse masks (optional alternative supervision) | **Santosh (`santosh@172.17.254.146`, conda env `ups`), Kuldeep hard drive (`/mnt/kuldeep` by default)** | RAMA's CUDA bindings do not run on Apple MPS. |

If RAMA-based coarse masks ever become an *alternative* supervision source
(the design above uses depth-CC coarse masks from
`generate_depth_guided_instances`, so RAMA is not required), launch them with:

```bash
bash scripts/remote_rama_santosh.sh train
# or, if Kuldeep is mounted elsewhere:
REMOTE_KULDEEP_ROOT=/media/kuldeep bash scripts/remote_rama_santosh.sh train
```

The script writes everything under `${REMOTE_KULDEEP_ROOT}/mbps_panoptic_segmentation/`
on Santosh, rsyncs the local RAMA scripts there, runs
`test-instance-labels/Superpixels/scripts/run_rama_official.py`, and pulls the
output NPZ back into `test-instance-labels/Superpixels/runs/rama_<split>/`
locally. Kuldeep is the chosen storage so the Santosh home partition stays
small.
