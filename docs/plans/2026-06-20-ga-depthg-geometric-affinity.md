# GA-DepthG — Geometric-Affinity DepthG (Track 1: clean ablation on DepthG's footing)

**Date:** 2026-06-20 · **Status:** spec (pre-implementation) · **Branch base:** `refs/depthg`
**Goal:** Replace DepthG's scalar depth-correlation loss term with a gravity-aligned ground-relative
(height + surface-normal) affinity, on DepthG's *own* frozen DINO ViT-B/8 footing, to make an
apples-to-apples "we beat DepthG" claim plus the required scalar→height→normal→both ablation.

Premise already proven (`reports/premise_check_geometry_affinity.md`): in the equal-depth regime,
geometry separates classes at AUROC 0.80 vs depth's 0.54; normals are the load-bearing cue; stereo
vs mono is a wash → keep mono. Novelty = partially-novel, claim only on role+setting
(`memory/novelty_geometric_affinity.md`).

## 1. The method (one-term swap)
DepthG term (verified `refs/depthg/src/modules.py:1256-1278`): `dd = depth_correlation(norm(d1), norm(d2))`,
loss `= -cd.clamp(0) * (dd - shift)`, added with weight `depth_feat_weight` to the STEGO loss.

Swap `dd` for a geometric affinity `A`:
- Per-pixel descriptor `g = [ĥ, n]` — `ĥ` = per-scene-standardized height-above-ground (scale-free →
  mono metric error irrelevant, per stereo finding); `n` = unit surface normal (3-ch).
- `A_ij = w_n · (n_i · n_j) + w_h · (ĥ_i · ĥ_j)` with `w_n > w_h` (start 0.6/0.4; normals load-bearing).
- `n_i·n_j` and `ĥ_i·ĥ_j` are each just `depth_correlation`'s einsum on the respective channels —
  **no new op needed**, and we BYPASS `norm()` for geometry (norm collapses scalar depth to sign — that
  degeneracy is why DepthG's depth weight is a tiny 0.09; our multi-channel affinity avoids it).

Train-only; inference stays image-only (monocular identity intact). Same backbone, sampling
(Cityscapes uses `depth_sampling=none` → random, so sampling is NOT a confound), train/eval as DepthG.

## 2. Exact integration points (surgical)
| change | file:line | what |
|---|---|---|
| new `geometric_affinity(g1,g2,mode,w_h,w_n)` | `src/modules.py` near :812 | normal-cosine + height term; `mode` selects ablation |
| use it in loss | `src/modules.py:1256` `depth_feature_correlation` | if `affinity_mode!='scalar'`: `dd = geometric_affinity(...)` else keep `depth_correlation` |
| thread geometry through forward | `src/modules.py:1280` `forward(... depth, depth_pos)` | pass `geom, geom_pos` (4-ch) when geometry mode on |
| load geometry | `src/data.py:179` depth block | add `return_geometry`/`geometry` type → load cached 4-ch `.npy` like depth |
| config keys | `src/configs/train_config.yml` | `affinity_mode, geom_w_h, geom_w_n, return_geometry` |
| guidance decay reuse | `src/depth_decay_modules.py` | reuse existing `depth_loss_decay` for the geom term |

## 3. New code: geometry features (mostly built)
`mbps_pytorch/premise_check_geometry_affinity.py` already has validated `back_project`,
`fit_ground_plane` (RANSAC IRLS), `surface_normals`. Refactor into `refs/depthg/src/geometry_features.py`:
- input: DepthPro inverse-depth npy + Cityscapes camera json (intrinsics, cam height).
- **fit ground plane on the FULL image (pre-crop)** for RANSAC stability, then apply DepthG's crop
  transform to the resulting height/normal maps (crop-alignment is the main data-prep risk).
- output per image: 4-ch tensor `[ĥ, n_x, n_y, n_z]`, `ĥ` per-scene-standardized
  `(h - median_ground)/scale_90`, `n` unit. Cache to disk in DepthG's data layout (parallel to `zoe_depth/`).

## 4. Ablation matrix (one knob = `affinity_mode`)
| run | `A_ij` | tests |
|---|---|---|
| `scalar` (matched internal baseline) | `depth_correlation(norm(d1),norm(d2))` on **DepthPro** | the clean within-pipeline baseline |
| `height` | `ĥ_i · ĥ_j` | height alone |
| `normal` | `n_i · n_j` | orientation alone |
| `both` (GA-DepthG) | `w_n(n_i·n_j)+w_h(ĥ_i·ĥ_j)` | the contribution |

**Depth consistency:** all four runs use the SAME depth (DepthPro), so scalar→both is internally clean.
DepthG's *published* 23.1 (ZoeDepth) is a separate EXTERNAL reference, reproduced via `cityscapes_vitb.ckpt`.
The honest claim is `both` > `scalar` (matched), with 23.1 cited as the external DepthG number.

## 5. Data prep
- Geometry cache for Cityscapes train+val crops, aligned to DepthG's `crop_datasets.sh` five-crop layout.
- Depth source: `depth_depthpro` (project) — note DepthG's own pipeline used ZoeDepth; we substitute
  DepthPro (already the project default; flag in paper as depth-model choice, not a contribution).

## 6. Eval
- Protocol: k-means cluster on head codes → CRF (`src/train_crf.py`) → Hungarian (`src/eval_segmentation.py`),
  27-class Cityscapes — identical to DepthG and CUPS. Report mIoU + Acc.
- **Baseline sanity FIRST:** eval `saved_models/cityscapes_vitb.ckpt` → expect ≈23.09 (locks the protocol
  before any training). Then each ablation run is trained + evaluated identically.

## 7. Env & compute
- No dedicated DepthG venv yet; `requirements.txt` is pinned (STEGO/pytorch-lightning era) → create
  `.venv_depthg` matching it. `.venv_cups_cpu` (torch 2.12) is likely too new.
- DINO ViT-B/8 backbone via `src/download_models.py`.
- **Train on remote GPU** (santosh / A6000) per project rule (training stays remote); **eval + geometry
  precompute local** (Mac/CPU). Geometry precompute is fast (150 imgs in seconds in the premise check).

## 8. Hyperparameters (from `paper_reproduction.sh` Cityscapes ViT-B)
`batch_size=32 dataset_name=cityscapes model_type=vit_base dim=100 depth_feat_correlation_loss=True
depth_feat_shift=0.03 depth_feat_weight=0.09 depth_loss_decay=True depth_loss_decay_factor=0.8
depth_sampling=none pointwise=False pos_intra=0.39/0.95 pos_inter=0.25/1.02 neg_inter=0.26/0.57`
New: `affinity_mode∈{scalar,height,normal,both} geom_w_h=0.4 geom_w_n=0.6`. Retune `depth_feat_shift`
(bias b) and `depth_feat_weight` (λ) for geometry — the scalar-tuned 0.09/0.03 likely under-weights a
non-degenerate affinity; small sweep.

## 9. Risks & kill-gate
- **Kill-gate:** if `both` does not beat DepthG `scalar` (23.1) on the identical protocol after a weight/
  shift sweep → the loss fails to convert the proven signal; stop or revisit. One training run reaches it.
- Normal noise from mono depth → soft affinity + normal weighting + guidance decay mitigate.
- Crop-alignment of geometry vs DepthG crops = main correctness risk (fit plane pre-crop).
- `depth_feat_weight`/`shift` retune needed (scalar values are tuned for the degenerate norm path).

## 10. Tasks
1. Create `.venv_depthg` from `requirements.txt`; download DINO ViT-B/8; reproduce baseline eval ≈23.1.
2. `geometry_features.py` (refactor premise-check geo; pre-crop ground fit; per-scene `ĥ`; 4-ch cache).
3. Cache geometry for Cityscapes train+val (crop-aligned).
4. `geometric_affinity()` + wire into `depth_feature_correlation` + `forward` + `data.py` + config.
5. Smoke: 1-step train with `affinity_mode=both` (shapes, no NaN); unit-test affinity (normal-cosine=1 for
   identical normals, =0 for orthogonal; height term monotone).
6. Run 4-way ablation (scalar/height/normal/both) on remote GPU; small `depth_feat_weight/shift` sweep on `both`.
7. Eval all (cluster+CRF+Hungarian, 27-class) → the ablation table; compare to DepthG 23.1.

## 11. Success criteria
- Baseline reproduced (≈23.1) → protocol locked.
- `both` > `scalar` by a clear margin; `normal` ≳ `height` (consistent with premise AUROCs).
- Clean ablation table for the paper; framing per `memory/novelty_geometric_affinity.md` (role+setting only).
