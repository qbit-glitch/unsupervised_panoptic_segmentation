# Mode B CAUSE-TR with DINOv3 ViT-B/16 + Frozen Codebook & Cluster Probe

## Context

CAUSE-TR's Cityscapes checkpoint releases two trained artifacts:

- `codebook (2048, 768)` — modularity prototypes in DINOv2 768-d feature space
- `cluster_probe (27, 90)` — 27-class centroids in TR decoder's 90-d output space

Both already extracted to `refs/cause/CAUSE/cityscapes/extracted_centroids/cause_tr_centroids_cityscapes.npz` (verified shapes; codebook is bit-identical to `modular.npy`).

We previously asked: "if we freeze these centroids and retrain CAUSE-TR with DINOv3, will we get similar results?" The answer is yes in expectation, with caveats. Mode B isolates the question — does the codebook's learned directional geometry transfer across DINO generations through nothing more than a small linear realignment?

Prior `CAUSE_dinov3_v1..v9` attempts in this repo never used the OFFICIAL DINOv2 codebook — they all retrained from scratch with smaller codebooks (512) or larger backbones (Large/16 with dim=1024). v5 was closest (frozen codebook) but used self-derived prototypes, not CAUSE's official ones. Mode B is the missing experiment.

**Outcome we want:** mIoU within ±1 of CAUSE-TR's published DINOv2 number on Cityscapes (27-class with Hungarian matching), proving feature-space transferability of the learned codebook.

**Why this matters for the broader project:** if Mode B works, future backbone swaps (DINOv4, etc.) reduce to "train one Linear adapter + TR decoder" instead of full pipeline retraining. If it fails, we learn the codebook geometry is backbone-specific and must be relearned.

---

## Approach

Stage-B-only training:
- **Frozen**: DINOv3 ViT-B/16 backbone, `codebook (2048, 768)`, `cluster_probe (27, 90)`
- **Trained from scratch**: TR decoder (`Segment_TR.head`), its EMA twin (`head_ema`, EMA-updated, not gradient-trained), `projection_head` + EMA twin, NEW `DINOv3ToDINOv2Adapter` (Linear 768→768, identity-init)
- **Skipped**: Stage A (modularity codebook learning) — not needed since codebook is loaded
- **Resolution**: 320×320 (smallest multiple of 16 near CAUSE's 322 → 20×20=400 patches; eval upsamples to label res)
- **Losses**: contrastive (codebook-bank, CAUSE constants) + centroid (custom, non-detached, since frozen probe has no grad path through CAUSE's `.detach()` version)

---

## Files to create

| Path | Purpose | LoC est. |
|------|---------|----------|
| `mbps_pytorch/models/adapters/dinov3_to_dinov2_adapter.py` | `DINOv3ToDINOv2Adapter` — Linear(768,768) identity-init | ~50 |
| `mbps_pytorch/training/cause_modeb_freeze.py` | `load_frozen_codebook_and_probe`, `install_frozen_into_cluster`, `verify_freeze` | ~120 |
| `mbps_pytorch/training/cause_modeb_trainer.py` | `CAUSEModeBTrainer` class — Stage-B-only loop, EMA, bank, frozen-aware centroid loss | ~280 |
| `mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml` | Single config: hyperparams, paths, freeze flags | ~70 |
| `mbps_pytorch/train_cause_modeb_dinov3.py` | CLI entry: arg parsing, model/dataset construction, trainer.fit() | ~200 |
| `mbps_pytorch/scripts/eval_cause_modeb.py` | Standalone eval (Cityscapes val, 27-class mIoU + Hungarian) | ~150 |
| `mbps_pytorch/tests/test_cause_modeb_freeze.py` | 10 sanity-check unit tests | ~250 |

## Files to read/import (no modification)

| Path | What we use |
|------|-------------|
| `mbps_pytorch/models/backbone/dinov3_vitb.py:259` | `DINOv3ViTB.from_pretrained` (frozen 768-d, strips CLS+4 registers) |
| `refs/cause/modules/segment.py:27-47` | `Segment_TR` class (head/head_ema, projection_head/ema, linear) |
| `refs/cause/modules/segment_module.py:77-90` | `Decoder` (codebook param, query_pos sized by `num_queries`, VQ + TRDecoder) |
| `refs/cause/modules/segment_module.py:92-235` | `Cluster` (codebook, cluster_probe, bank ops, contrastive_ema_with_codebook_bank) |
| `refs/cause/modules/segment_module.py:342-347` | `ema_init`, `ema_update` |
| `mbps_pytorch/train_cause_dinov3.py:294-331` | `DeviceAwareCluster` (MPS/CPU-safe replacement for `.cuda()`-hardcoded bank ops in `Cluster.bank_init`/`bank_compute` at segment_module.py:114, 139) |
| `mbps_pytorch/train_cause_dinov3.py:481-614` | Cityscapes val dataset + Hungarian-matched eval |
| `refs/cause/CAUSE/cityscapes/extracted_centroids/cause_tr_centroids_cityscapes.npz` | Frozen artifacts (already extracted) |

---

## Critical correctness fixes (from code verification)

### Fix 1: `forward_centroid` detaches input → zero gradient with frozen probe

`refs/cause/modules/segment_module.py:221` does `transform(x.detach())`. In CAUSE, `cluster_probe` is trainable, so gradient flows through `normed_clusters`. **Mode B freezes `cluster_probe`**, so the entire centroid loss has zero gradient. Must replace with a custom non-detached version inside the trainer:

```python
# Inside CAUSEModeBTrainer
def _centroid_loss_modeb(self, seg_feat: Tensor) -> Tensor:
    """Variant of Cluster.forward_centroid WITHOUT .detach() — flows grad
    into TR decoder and adapter. cluster_probe stays frozen via requires_grad=False."""
    normed_features = F.normalize(transform(seg_feat), dim=1)             # NO .detach()
    normed_clusters = F.normalize(self.cluster.cluster_probe, dim=1)      # frozen
    inner = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)
    one_hot = F.one_hot(inner.argmax(1), self.cluster.cluster_probe.shape[0]) \
                .permute(0, 3, 1, 2).float()
    return -(one_hot * inner).sum(1).mean()
```

Pass **student** `seg_feat` (not `seg_feat_ema`) so the head trains.

### Fix 2: `Decoder.codebook` is `None` by default

`Segment_TR.__init__` calls `Decoder(args)` without codebook (segment.py:38). `Decoder.__init__` defaults to `codebook=None` (segment_module.py:78), but `Decoder.forward` does `vqt(feat, self.codebook)` (segment_module.py:87). Must wire after construction:

```python
segment.head.codebook       = cluster.codebook   # both point to same frozen Parameter
segment.head_ema.codebook   = cluster.codebook
```

This is also why `install_frozen_into_cluster` must overwrite `cluster.codebook` *before* the segment is constructed, OR re-wire after both exist.

### Fix 3: `query_pos` size

`Decoder.query_pos = nn.Parameter(torch.randn(args.num_queries, args.dim))` (segment_module.py:83). Set `args.num_queries = 400` (= 20×20 patches) to match DINOv3 @ 320 input. Trained from scratch — no need to interpolate the released DINOv2 query_pos (which was 529 at 23×23).

### Fix 4: Bank ops hardcoded to `.cuda()`

`Cluster.bank_init/bank_compute` at segment_module.py:114, 139 call `.cuda()` directly. **Use `DeviceAwareCluster` from `mbps_pytorch/train_cause_dinov3.py:294-331`** — already verified MPS/CUDA/CPU safe.

---

## Module signatures

### `DINOv3ToDINOv2Adapter` — `mbps_pytorch/models/adapters/dinov3_to_dinov2_adapter.py`

```python
class DINOv3ToDINOv2Adapter(nn.Module):
    """Linear(768→768) initialized to identity. Trainable.
    No EMA twin: the same instance serves student and teacher paths
    so they stay in lockstep (the codebook downstream is frozen and
    cannot absorb a student/teacher adapter mismatch).
    Param count: 590,592 (768*768 + 768).
    """
    def __init__(self, dim: int = 768, init: str = "identity"):
        super().__init__()
        self.fc = nn.Linear(dim, dim, bias=True)
        if init == "identity":
            nn.init.eye_(self.fc.weight)
            nn.init.zeros_(self.fc.bias)
    def forward(self, x: Tensor) -> Tensor:  # (B, N, 768) -> (B, N, 768)
        return self.fc(x)
```

### `CAUSEModeBTrainer` — `mbps_pytorch/training/cause_modeb_trainer.py`

```python
class CAUSEModeBTrainer:
    def __init__(self, *, backbone, adapter, segment, cluster, device, cfg): ...
    @torch.no_grad()
    def _backbone_features(self, img) -> Tensor: ...      # (B, 400, 768)
    def _step(self, img) -> Dict[str, Tensor]: ...        # returns {loss, contrastive, centroid}
    def _ema_step(self) -> None: ...                       # head + projection_head only
    @torch.no_grad()
    def _bank_update(self, adapted_feat, proj_feat_ema) -> None: ...
    def _centroid_loss_modeb(self, seg_feat) -> Tensor: ...  # see Fix 1
    def fit(self, train_loader, val_loader) -> None: ...
    def validate(self, val_loader) -> Dict[str, float]: ...  # mIoU, pAcc
```

---

## Configuration — `mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml`

```yaml
experiment_name: cause_modeb_dinov3_vitb16_cs27
seed: 42

backbone:
  model_name: facebook/dinov3-vitb16-pretrain-lvd1689m
  freeze: true
  embed_dim: 768
  patch_size: 16

resolution: 320              # 320/16 = 20 → 400 patches; smallest mult-of-16 near CAUSE 322
patches_per_side: 20
num_patches: 400

frozen_centroids_npz: refs/cause/CAUSE/cityscapes/extracted_centroids/cause_tr_centroids_cityscapes.npz

cause:
  dim: 768
  reduced_dim: 90
  projection_dim: 2048
  num_codebook: 2048           # MUST match frozen codebook shape
  n_classes: 27                # MUST match frozen cluster_probe rows
  num_queries: 400             # = num_patches; trained from scratch

adapter:
  type: linear_identity_init
  in_dim: 768
  out_dim: 768
  bias: true

training:
  epochs: 40                   # CAUSE Stage B = 40 epochs (refs/cause/train_cause_tr_dinov2.py:533)
  batch_size: 8                # A6000 48GB; drop to 2 on 1080 Ti, 4 on M4 Pro MPS
  grad_accum_steps: 2          # effective bs=16
  optimizer: adam
  head_lr: 5.0e-5              # = CAUSE original (refs/cause/train_cause_tr_dinov2.py:534)
  adapter_lr: 1.0e-4           # 2× head_lr; small enough not to dominate
  weight_decay: 0.0
  grad_clip_norm: 1.0
  ema_decay: 0.99
  contrastive_temp: 0.07
  pos_thresh: 0.3
  neg_thresh: 0.1
  bank_max_size: 100
  loss_weights: {contrastive: 1.0, centroid: 1.0}

validation:
  every_epoch: 1
  flip_tta: true
  hungarian_match: true        # 27×27, identity-like once converged

data:
  cityscapes_root: /Users/qbit-glitch/Desktop/datasets/cityscapes
  num_workers: 4
  imagenet_mean: [0.485, 0.456, 0.406]
  imagenet_std:  [0.229, 0.224, 0.225]

output_dir: refs/cause/CAUSE_modeb_dinov3_vitb16
save_every: 5
log_every: 50
```

---

## Training procedure

### Init
1. Load YAML.
2. `backbone = DINOv3ViTB.from_pretrained(cfg.backbone.model_name).to(device).eval()`. All params already `requires_grad=False` per `dinov3_vitb.py:340`.
3. `cb, cp = load_frozen_codebook_and_probe(cfg.frozen_centroids_npz, device)`. Both `requires_grad=False`.
4. `args = SimpleNamespace(dim=768, reduced_dim=90, projection_dim=2048, num_codebook=2048, n_classes=27, num_queries=400)`.
5. `cluster = DeviceAwareCluster(args, device)`. Then `install_frozen_into_cluster(cluster, cb, cp)` — replaces `cluster.codebook` and `cluster.cluster_probe` with frozen `nn.Parameter(..., requires_grad=False)`.
6. `cluster.bank_init()` (DeviceAware version uses correct device).
7. `segment = Segment_TR(args).to(device)`. Wire codebook: `segment.head.codebook = cluster.codebook; segment.head_ema.codebook = cluster.codebook`.
8. `ema_init(segment.head, segment.head_ema)`; `ema_init(segment.projection_head, segment.projection_head_ema)`.
9. `adapter = DINOv3ToDINOv2Adapter(dim=768, init="identity").to(device)`.
10. Build optimizer with two param groups:
    - **Group A** (lr=5e-5): all params of `segment.head` (excludes its `.codebook` since that's the frozen ref), `segment.projection_head`, `segment.linear`.
    - **Group B** (lr=1e-4): `adapter.parameters()`.
    - Excluded from optimizer: backbone, all EMA params, `cluster.codebook`, `cluster.cluster_probe`.
11. Run `verify_freeze(...)` — assert `requires_grad=False` on backbone params, codebook, cluster_probe; assert codebook/probe values match the loaded npz.

### Per batch
1. `img = batch["img"].to(device)` → (B, 3, 320, 320).
2. `with torch.no_grad(): raw_feat = backbone(img)` → (B, 400, 768) (CLS+4 registers stripped per `dinov3_vitb.py:222-226`).
3. `adapted_feat = adapter(raw_feat)` — gradient flows.
4. **Student**: `seg_feat = segment.head(adapted_feat, drop=segment.dropout)` → (B, 400, 90); `proj_feat = segment.projection_head(seg_feat)` → (B, 400, 2048).
5. **Teacher** (`with torch.no_grad():`): `seg_feat_ema = segment.head_ema(adapted_feat); proj_feat_ema = segment.projection_head_ema(seg_feat_ema)`.
6. `cluster.bank_compute()` — refresh.
7. `loss_contrastive = cluster.contrastive_ema_with_codebook_bank(adapted_feat, proj_feat, proj_feat_ema, temp=0.07, pos_thresh=0.3, neg_thresh=0.1)`.
8. `loss_centroid = self._centroid_loss_modeb(seg_feat)` (custom, non-detached — Fix 1).
9. `loss = loss_contrastive + loss_centroid`. Backward.
10. `clip_grad_norm_(trainable_params, 1.0)`; `optimizer.step()`; `optimizer.zero_grad(set_to_none=True)`.
11. `ema_update(segment.head, segment.head_ema, 0.99)`; same for projection_head.
12. `with torch.no_grad(): cluster.bank_update(adapted_feat, proj_feat_ema, max_num=100)`.

### Per epoch
- `gc.collect(); torch.{cuda,mps}.empty_cache()`.
- Save checkpoint every `save_every` epochs: `{epoch_NNN}/{segment_tr.pth, cluster_tr.pth, adapter.pth}` plus `train_meta.json`.
- Validate every epoch; if `mIoU > best`, save under `best/`.

---

## Evaluation procedure

Lift `validate()` from `mbps_pytorch/train_cause_dinov3.py:541-614` with two changes:
1. Insert `adapted = adapter(raw_feat)` between backbone forward and `segment.head_ema(...)`.
2. `n_cluster = 27` (was 54 in v9).

Steps:
1. For each val image: ImageNet-norm; resize to 320; center-crop 320.
2. Backbone forward (no_grad) on `img` and `img.flip(dims=[3])` → 2 × (B, 400, 768).
3. `adapter(...)` — still 768-d.
4. `segment.head_ema(...)` on each → (B, 400, 90); reshape to (B, 90, 20, 20).
5. Average orig + horizontal-flipped-back → (B, 90, 20, 20).
6. Bilinear upsample to (1024, 2048) — Cityscapes label resolution.
7. Cosine sim vs `cluster.cluster_probe (27, 90)`; argmax → predicted class indices.
8. Accumulate 27×27 confusion against GT (27-class CAUSE label remapping).
9. After full pass: `linear_sum_assignment(conf, maximize=True)` → row→col permutation.
10. mIoU over mapped pairs; pAcc = trace/total.

**Expected**: mIoU 22–24 @ epoch 5; 26–28 @ epoch 20; final 28–30 (CAUSE-TR DINOv2 published is ~28 on this metric). Target: within ±1 of CAUSE-TR DINOv2.

---

## Sanity tests (must all pass before 40-epoch run)

In `mbps_pytorch/tests/test_cause_modeb_freeze.py`:

1. `codebook.shape == (2048, 768)`, `cluster_probe.shape == (27, 90)`, both float32, both `requires_grad=False`.
2. After model assembly, every backbone param has `requires_grad=False`; same for `cluster.codebook` and `cluster.cluster_probe`.
3. After 1 forward+backward+`optimizer.step()`, `torch.equal(cb_before, cluster.codebook)` and same for `cluster_probe`.
4. Adapter weight has changed from identity after 1 step: `torch.norm(adapter.fc.weight - eye) > 0`.
5. At step 0, `torch.allclose(adapter(x), x, atol=1e-6)` — identity init confirmed.
6. Single-batch overfit: 10 iters on a single 8-image batch → loss drops > 50%. (If not, gradients are not flowing — likely Fix 1 wasn't applied correctly.)
7. Shape pipeline: backbone → (B,400,768) → adapter → (B,400,768) → head → (B,400,90) → projection_head → (B,400,2048).
8. Save+load roundtrip: codebook and cluster_probe bitwise-identical.
9. Across 2 epochs of training, `cluster_probe` rows are unchanged at every checkpoint.
10. Pre-training mIoU on a 100-image subset > 5% — rules out a class-mapping bug (random predictions would give ~3.7%).

---

## Risk and abort criteria

| # | Symptom | Fallback |
|---|---------|----------|
| 1 | Sanity #6 fails (overfit loss doesn't drop) | Confirm Fix 1 applied — `_centroid_loss_modeb` does NOT call `cluster.forward_centroid`. Confirm input is student `seg_feat`, not `seg_feat_ema`. |
| 2 | mIoU @ epoch 5 < 15 | DINOv3↔codebook gap too large for Linear adapter. Promote to 2-layer MLP `Linear(768,768)→GELU→Linear(768,768)`, identity-init both layers (second layer init.eye, first init.eye + bias=0). Re-run from epoch 0. |
| 3 | mIoU @ epoch 10 < 22 | MLP adapter still insufficient. **Unfreeze `cluster_probe`** at lr=1e-5 (codebook stays frozen). |
| 4 | mIoU @ epoch 15 < 24 with unfrozen probe | Codebook geometry itself doesn't transfer. **Unfreeze codebook at lr=1e-6** as soft prior. This is the last fallback before declaring Mode B infeasible. |
| 5 | NaN / loss explosion | Reduce `head_lr` to 1e-5; clip grad to 0.5; `pos_thresh=0.2`. |
| 6 | Bank starves (`flat_norm_bank_proj_feat_ema.shape[0] < num_codebook`) for >5 epochs | Increase `bank_max_size` to 200; reduce `bank_update` random-cut ratio from 0.5 to 0.7. |

---

## End-to-end verification

```bash
cd /Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation

# 1) Sanity tests (must pass before any training)
python -m pytest mbps_pytorch/tests/test_cause_modeb_freeze.py -v

# 2) Smoke run (1 epoch, 8 batches, ~2 min)
python mbps_pytorch/train_cause_modeb_dinov3.py \
    --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
    --epochs 1 --max_batches 8 --device auto

# 3) Full training (40 epochs)
nohup python -u mbps_pytorch/train_cause_modeb_dinov3.py \
    --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
    --device auto \
    > logs/cause_modeb_dinov3_train.log 2>&1 &

# 4) Eval on best checkpoint
python mbps_pytorch/scripts/eval_cause_modeb.py \
    --config mbps_pytorch/configs/cause_modeb_dinov3_vitb16.yaml \
    --checkpoint refs/cause/CAUSE_modeb_dinov3_vitb16/best
```

**Wall-clock & memory estimates** (2975 train images, 40 epochs ≈ 14,880 steps):

| Hardware | Batch | Peak mem | Time/epoch | Total |
|----------|-------|----------|------------|-------|
| A6000 48GB | 8 | ~6.5 GB | ~18 min | ~12 h |
| GTX 1080 Ti 11GB | 2 (accum=4) | ~9.5 GB | ~45 min | ~30 h |
| M4 Pro 48GB MPS | 4 | ~7 GB | ~38 min | ~25 h |

**Success criterion**: final mIoU within ±1 of CAUSE-TR DINOv2 published number on Cityscapes 27-class with Hungarian matching. If achieved, this is a clean feature-space-reproducibility result demonstrating that learned codebook geometry transfers across DINO generations through linear realignment alone.

---

## Open question for confirmation before implementing

The plan assumes you want to ship Mode B as a **standalone reproducibility experiment** (clean comparison with CAUSE-TR DINOv2 baseline). If instead you want this as a **building block for the broader MBPS pipeline** (e.g., to produce DINOv3-based 27-class semantic pseudo-labels for downstream Cascade Mask R-CNN training), the eval and checkpoint format need an extra step: export per-image `(H, W)` predictions as PNGs in CUPS-compatible format. Easy to add — flag as a yes/no during implementation kickoff.
