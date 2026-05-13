# Plan — Fix LoRA/DoRA Adapter Architecture for DINOv2 + CAUSE-TR Stage-1

## Context

The current Stage-1 adapter training (DINOv2 ViT-B/14 + CAUSE-TR head) is failing in three correlated ways:

1. **Numerical instability** — DoRA's column-norm produced `loss=inf` spikes in E2 (steps 528, 770, 5278, 8304, 8985, 10585). Root cause: `V / ‖V‖` in `DoRALinear.forward()` at `mbps_pytorch/models/adapters/lora_layers.py:114-115` is brittle under gradients from a weak distillation signal.
2. **Zero-gradient init hack** — `lora_B=0` combined with cosine distillation gives `1-cos(0)=0` with zero gradient. Worked around by a `std=0.01` Gaussian perturbation at `mbps_pytorch/train_semantic_adapter.py:497-501` (commit C020). This is a band-aid, not a principled init.
3. **Wrong training signal** — feature-cosine distillation against the frozen teacher incentivizes `student = teacher` (i.e., adapter does nothing). DA3 run (C021) confirmed: val MAE best at epoch 1 (0.0131), degraded to 0.0443 by epoch 50. The adapter overfits the distillation objective and then drifts away from useful features.
4. **Wasteful target surface** — tiered injection adapts `attn.qkv` (fused), `attn.proj`, `mlp.fc1`, `mlp.fc2` across 12 blocks. Adapting K entangles Q/K/V updates (fused qkv shares rank), FFN drift drives overfitting, and full-depth adaptation wastes capacity on early blocks that don't need it.

Evidence that adaptation can work — just not under the current setup: Conv-DoRA on the DepthPro (cleaner) pseudo-label regime achieved PQ=28.28 vs frozen 28.40, closing 92% of the gap (`memory/e2_conv_dora_results.md`). The adapter architecture is signal-limited, not fundamentally broken.

**Intended outcome.** A staged three-experiment rollout that (a) stabilizes training (no `loss=inf`, no init hacks), (b) narrows the adapter surface to the last four transformer blocks on Q/V only, and (c) replaces feature-cosine distillation with a code-space SwAV objective initialized from the existing k=80 k-means centroids — an objective the adapter can actually improve against. Each experiment is independently verifiable so gains/regressions can be attributed.

---

## Shared preconditions (apply to all three experiments)

- Working dir: `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation`
- Branch: `implement-dora-adapters` (current)
- Downstream loaders already tolerate new `adapter_config` fields via `.get()` — verified at `mbps_pytorch/generate_semantic_pseudolabels_adapted.py:253-288` and `mbps_pytorch/eval_cause_k80.py:331-337`. No changes to those scripts are needed for backward compatibility.

---

## Exp 1 — LoRA + PiSSA init (lowest-risk refactor)

**Goal.** Eliminate `loss=inf` spikes and remove the `std=0.01` symmetry-breaking hack. Must match the frozen baseline (no regression) — this alone settles whether the architecture is stable before changing the target surface or the loss.

### Files to modify

1. **`mbps_pytorch/models/adapters/lora_layers.py`** — add a module-level `pissa_init_(adapter: LoRALinear, n_iter: int = 7)` helper after `LoRALinear` (line 67, before `DoRALinear`).
   - Runs `torch.svd_lowrank(W, q=rank + n_iter, niter=n_iter)` on the frozen weight (`self.weight` buffer at line 40).
   - Sets `lora_A = sqrt(S_r) · V_rᵀ` (shape `(r, in_features)`), `lora_B = U_r · sqrt(S_r)` (shape `(out_features, r)`).
   - Recomputes residual: `self.weight.data = W - B @ A`. After this, `forward()` at step 0 reproduces the original output exactly (forward-preserving property), but with non-zero `lora_A`/`lora_B` that give immediate gradient signal.
   - Handles the scaling convention: since `LoRALinear.forward` at line 61 multiplies by `scaling = alpha/r`, either (a) inject `1/scaling` into `lora_B` so the arithmetic cancels, or (b) set `self.scaling = 1.0` when PiSSA-initialized. Use (b) — cleaner and traceable via a `self.pissa_initialized: bool` flag.

2. **`mbps_pytorch/train_semantic_adapter.py`** — replace the symmetry-breaking block at lines 493-501:
   - If `args.pissa_init and args.variant == "lora"`: iterate `backbone.named_modules()` and `segment.named_modules()`, call `pissa_init_(mod)` on every `LoRALinear`.
   - Else: keep the existing `std=0.01` perturbation (legacy path, still needed for `variant="dora"` and legacy checkpoints).
   - Error out early if user combines `--variant dora --pissa_init` (PiSSA requires the linear LoRA math, not DoRA's magnitude decomposition).

3. **`mbps_pytorch/train_semantic_adapter.py:388,393`** — add CLI flag `--pissa_init` (action="store_true") and change `--variant` default from `"dora"` to `"lora"`. Extend `adapter_config` dict at lines 625-632 with `"pissa_init": args.pissa_init`.

4. **`mbps_pytorch/tests/test_dora_adapter_training.py`** — append two tests after `test_ema_head_adapted`:
   - `test_pissa_init_preserves_output()` — verifies forward output equals the pre-PiSSA output within `atol=1e-4`.
   - `test_pissa_nonzero_gradient_at_init()` — verifies `lora_A.grad` and `lora_B.grad` are non-zero after one backward with a perturbed teacher feature (proves no flat-region start).

### Cross-cutting regimen fixes (apply with Exp 1)

These are cheap config changes that address the DA3 overfitting pattern; include in Exp 1 so it's a fair comparison.

- **Default epochs: 50 → 5** (`train_semantic_adapter.py:386`).
- **Default LR: 1e-4 → 5e-5** (`train_semantic_adapter.py:387`).
- **Warmup + cosine schedule**: replace `CosineAnnealingLR(T_max=epochs)` at line 279 with a `LambdaLR` that does linear warmup over `--warmup_steps 1000` (new CLI flag) then cosine decay over the remaining steps. Move `scheduler.step()` from line 358 to per-batch (after `optimizer.step()` at line 345).
- **Early-stop on downstream mIoU**: the current best-checkpoint save at lines 368-375 triggers on training-loss minimum, which rewards converging to "do nothing". Add an inline evaluation hook that runs on ~100 val images at each epoch and saves `best_val_miou.pt` based on k=80 mIoU. New CLI flags `--eval_every 1`, `--eval_subset_size 100`. Reuse the k-means + Hungarian pipeline from `mbps_pytorch/eval_cause_k80.py:521-613` (`evaluate_kmeans`).

### Verification (Exp 1)

Local macOS MPS smoke run (≤5 min):
```
python mbps_pytorch/train_semantic_adapter.py \
  --data_dir /Users/qbit-glitch/Desktop/datasets \
  --output_dir /tmp/smoke_exp1 \
  --variant lora --pissa_init \
  --rank 4 --alpha 4.0 --late_block_start 6 \
  --losses distillation --loss_weights '{"distillation": 1.0}' \
  --epochs 1 --batch_size 2 --lr 5e-5 --warmup_steps 50 \
  --eval_every 0 --save_every 1
```

Pass criteria:
- No `inf` or `nan` in log.
- Checkpoint loads in `eval_cause_k80.py` via `bash scripts/eval_dora_vs_frozen_k80_local.sh` without missing/unexpected key errors.

Full remote run on 2x1080Ti (replace `scripts/train_dino_adapter_distill.sh` body):
```
torchrun --nproc_per_node=2 mbps_pytorch/train_semantic_adapter.py \
  --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" \
  --variant lora --pissa_init --rank 4 --alpha 4.0 \
  --late_block_start 6 --losses distillation \
  --epochs 5 --lr 5e-5 --warmup_steps 1000 \
  --batch_size 16 --eval_every 1 --eval_subset_size 100 --save_every 1 --seed 42
```

Expected result: k=80 mIoU on Cityscapes val ≥ frozen baseline `52.69%` within ±0.2pp. No `inf` spikes across the 5-epoch run. **If this holds, Exp 1 has done its job** (stabilized training); any further improvement requires Exp 2 or Exp 3.

---

## Exp 2 — Split-QKV + drop FFN + late-4-only

**Goal.** Narrow the adapter surface. Target `(Q, V, proj)` in the last 4 blocks (indices 8..11) only. Trainable param count drops ~6× (463k → ~73k).

### Files to modify

1. **`mbps_pytorch/models/adapters/lora_layers.py`** — add a new class `LoRASplitQKV(nn.Module)` after `LoRAConv2d` (around line 236).
   - Wraps the fused `attn.qkv = nn.Linear(768, 2304)` (confirmed as fused at `refs/cause/models/dinov2vit.py:203`; output layout Q|K|V along dim=-1 per the reshape at line 210).
   - Constructor: `__init__(wrapped, rank, alpha, dropout, targets=("q","v"))`. For each target, create `lora_A_{t}` (r, 768) and `lora_B_{t}` (768, r) as first-class `nn.Parameter`s (not wrapped via list-of-params, so PiSSA can access them by name).
   - Forward: compute `base_out = F.linear(x, W)` → (B, N, 2304). For each target chunk `t ∈ targets`, compute delta `= scaling · B_t @ (A_t @ x)`, add into `base_out[..., idx*768:(idx+1)*768]` (idx=0 for q, 1 for k, 2 for v). Return.
   - Include `trainable_count()` and a PiSSA path: `pissa_init_` must slice `W[idx*768:(idx+1)*768, :]` per active target and SVD each chunk separately, then recompute the residual `W_chunk - B_t @ A_t` in-place for the active chunks only.

2. **`mbps_pytorch/models/adapters/lora_layers.py:297-309`** — extend `freeze_non_adapter_params` suffix list to include `.lora_A_q`, `.lora_B_q`, `.lora_A_k`, `.lora_B_k`, `.lora_A_v`, `.lora_B_v`.

3. **`mbps_pytorch/models/adapters/dinov2_adapter.py:53-126`** — extend `inject_lora_into_dinov2` signature:
   ```
   attention_mode: str = "qkv_fused"      # new: "qkv_fused" | "q_v_split"
   qv_targets: Tuple[str, ...] = ("q","v")  # new
   adapt_ffn: bool = False                 # new: default False (was implicitly True for late blocks)
   ```
   Refactor the target-selection block at lines 97-118:
   - Skip early blocks entirely (no adaptation when `block_idx < late_block_start`).
   - In late blocks: if `attention_mode == "q_v_split"`, replace `block.attn.qkv` with a `LoRASplitQKV` wrapper; else wrap `attn.qkv` with the selected `adapter_cls`. Always wrap `attn.proj`. Wrap `mlp.fc1/fc2` only when `adapt_ffn=True`.

4. **`mbps_pytorch/train_semantic_adapter.py`** — add CLI flags next to existing adapter flags near line 393:
   ```
   --attention_mode {qkv_fused, q_v_split}  (default "qkv_fused")
   --qv_targets q,v                          (comma-separated)
   --adapt_ffn                               (action store_true, default False)
   ```
   Change `--late_block_start` default from `6` to `8` (line 392).
   Change `--adapt_cause` default to False (line 393) — drop CAUSE-TR head adaptation as part of the narrow philosophy.
   Extend `adapter_config` at lines 625-632 with `attention_mode`, `qv_targets`, `adapt_ffn` entries.
   Pass new args through to `inject_lora_into_dinov2` call at line 487.

5. **`mbps_pytorch/generate_semantic_pseudolabels_adapted.py:281-288`** and **`mbps_pytorch/eval_cause_k80.py:347-354`** — read new fields from `adapter_config` with backward-compatible defaults:
   ```
   cfg_attention_mode = cfg.get("attention_mode", "qkv_fused")
   cfg_qv_targets = tuple(cfg.get("qv_targets", "q,v").split(","))
   cfg_adapt_ffn = cfg.get("adapt_ffn", True)   # True for legacy checkpoints
   ```
   The `adapt_ffn=True` default at loading time keeps legacy checkpoints (trained with FFN adapters) loadable; new checkpoints store the explicit `False`.

6. **`mbps_pytorch/tests/test_dora_adapter_training.py`** — append three tests:
   - `test_lora_split_qkv_forward_preserving_after_pissa` — PiSSA-initialize split wrapper, verify forward output matches fused-weight baseline.
   - `test_lora_split_qkv_only_q_v_receive_grad` — only Q and V grads should be non-zero when `targets=("q","v")`; no K parameters should exist.
   - `test_late_4_injection_count` — with `late_block_start=8, adapt_ffn=False`, exactly 4 blocks should have adapters (Q, V, proj per block).

### Verification (Exp 2)

Local smoke run:
```
python mbps_pytorch/train_semantic_adapter.py \
  --data_dir /Users/qbit-glitch/Desktop/datasets \
  --output_dir /tmp/smoke_exp2 \
  --variant lora --pissa_init \
  --rank 4 --alpha 4.0 \
  --late_block_start 8 --attention_mode q_v_split --qv_targets q,v \
  --losses distillation --loss_weights '{"distillation": 1.0}' \
  --epochs 1 --batch_size 2 --lr 5e-5 --warmup_steps 50
```

Pass criteria:
- "Total trainable params" log at `train_semantic_adapter.py:276` reports ~73k (not ~463k).
- Forward pass produces finite losses. Checkpoint load round-trips.

Full run: same command with `--epochs 5 --eval_every 1`. k=80 mIoU should remain within ±0.5pp of frozen baseline (still no loss upgrade at this stage — we're only verifying the narrower surface doesn't regress).

---

## Exp 3 — Code-space SwAV loss with k=80 prototype init

**Goal.** Replace feature-cosine distillation with a SwAV objective on the 90D CAUSE codes. Initialize SwAV prototypes from the existing k=80 k-means centroids (file on disk: `pseudo_semantic_raw_k80/kmeans_centroids.npz`, shape `(80, 90)`). This is the first experiment where we expect to *beat* the frozen baseline.

### Files to modify

1. **`mbps_pytorch/losses/code_swav_loss.py`** — new file.
   - Port `sinkhorn_knopp_teacher` from `refs/dinov3/dinov3/loss/dino_clstoken_loss.py:43-70` as a standalone single-GPU version (drop the `dist.all_reduce` calls; keep the 3-iteration alternating-normalization body). Wrap with `@torch.no_grad()`.
   - Implement `code_swav_loss(code_student, prototypes, code_teacher=None, sinkhorn_iters=3, teacher_temp=0.05, student_temp=0.1, lambda_entropy=0.1) -> Tensor`:
     1. L2-normalize `code_student`, `code_teacher`, `prototypes` along the feature dim.
     2. Student logits `s_s = code_student @ prototypes.T` and teacher logits `s_t = code_teacher @ prototypes.T` (teacher detached or use augmented-view teacher).
     3. Under `@torch.no_grad()`: `q_t = sinkhorn_knopp(s_t, teacher_temp, sinkhorn_iters)`.
     4. Student `log_p_s = F.log_softmax(s_s / student_temp, dim=-1)`.
     5. SwAV cross-entropy: `-(q_t * log_p_s).sum(-1).mean()`.
     6. Entropy regularizer (prevents collapse): compute batch-mean `p̄ = p_s.mean(0)`, `H = -(p̄ * log(p̄ + 1e-8)).sum()`. Add `-lambda_entropy * H` (negative so we maximize H). Pattern adapted from `mbps_pytorch/losses/consistency_loss.py:19-70`.

2. **`mbps_pytorch/train_semantic_adapter.py`** — around line 547 (after cluster module is set up), load the k=80 prototypes:
   ```
   if "code_swav" in loss_list:
       centroids_path = os.path.join(args.data_dir, "cityscapes",
                                     "pseudo_semantic_raw_k80", "kmeans_centroids.npz")
       cents = np.load(centroids_path)["centroids"]  # (80, 90)
       swav_prototypes = nn.Parameter(
           torch.from_numpy(cents).float().to(device), requires_grad=True,
       )
   ```
   Register `swav_prototypes` as a new trainable parameter group alongside adapter params at line 272. Include it in checkpoints in the save blocks at lines 363-365 and 372-375.

3. **`mbps_pytorch/train_semantic_adapter.py:307-335`** — insert a new loss branch after the `cause_cluster` deprecated block:
   ```
   if "code_swav" in losses:
       from mbps_pytorch.losses.code_swav_loss import code_swav_loss
       code_s = transform(seg_feat_student)                     # (B, 90, H, W)
       code_t = transform(seg_feat_teacher).detach()            # (B, 90, H, W)
       B_, C_, H_, W_ = code_s.shape
       code_s_flat = code_s.permute(0, 2, 3, 1).reshape(-1, C_)
       code_t_flat = code_t.permute(0, 2, 3, 1).reshape(-1, C_)
       l_swav = code_swav_loss(
           code_s_flat, swav_prototypes,
           code_teacher=code_t_flat,
           sinkhorn_iters=args.sinkhorn_iters,
           teacher_temp=args.swav_teacher_temp,
           student_temp=args.swav_student_temp,
           lambda_entropy=args.swav_entropy_weight,
       )
       w = loss_weights.get("code_swav", 1.0)
       loss_total = loss_total + w * l_swav
       totals["code_swav"] += l_swav.item()
   ```
   Reuse `transform` already imported at line 46.

4. **`mbps_pytorch/train_semantic_adapter.py:393`** — add CLI flags:
   ```
   --sinkhorn_iters        int   default 3
   --swav_teacher_temp    float default 0.05
   --swav_student_temp    float default 0.1
   --swav_entropy_weight  float default 0.1
   ```

5. **`mbps_pytorch/tests/test_code_swav_loss.py`** — new file with three tests:
   - `test_sinkhorn_balanced_assignment` — feed random (256, 80) similarity matrix, 3 iters, assert row sums ≈ 1 and column sums ≈ 256/80.
   - `test_code_swav_loss_decreases_with_aligned_codes` — student=teacher=same random codes → small loss; student=random teacher=random → larger loss.
   - `test_code_swav_entropy_prevents_collapse` — when student predicts the same cluster for all pixels, the entropy-penalty component dominates.

### Default loss recipe for Exp 3 runs

```
--losses code_swav,distillation \
--loss_weights '{"code_swav": 1.0, "distillation": 0.1}'
```

Distillation drops from 1.0 → 0.1 (weak regularizer only). SwAV becomes primary.

### Verification (Exp 3)

Local smoke run with synthetic centroids (skip real file loading for the test):
```
# first, create a stub centroids file in smoke run data dir (1-liner)
python -c "import numpy as np; np.savez('/tmp/cs_smoke/pseudo_semantic_raw_k80/kmeans_centroids.npz', centroids=np.random.randn(80,90).astype('float32'))"

python mbps_pytorch/train_semantic_adapter.py \
  --data_dir /tmp/cs_smoke --output_dir /tmp/smoke_exp3 \
  --variant lora --pissa_init --rank 4 --alpha 4.0 \
  --late_block_start 8 --attention_mode q_v_split --qv_targets q,v \
  --losses code_swav,distillation \
  --loss_weights '{"code_swav": 1.0, "distillation": 0.1}' \
  --epochs 1 --batch_size 2 --lr 5e-5 --warmup_steps 50
```

Pass criteria:
- SwAV loss appears in totals dict and decreases over steps.
- No NaN/Inf.

Full run (on remote, using real k=80 centroids file):
```
torchrun --nproc_per_node=2 mbps_pytorch/train_semantic_adapter.py \
  --data_dir "$DATA_DIR" --output_dir "$OUTPUT_DIR" \
  --variant lora --pissa_init --rank 4 --alpha 4.0 \
  --late_block_start 8 --attention_mode q_v_split --qv_targets q,v \
  --losses code_swav,distillation \
  --loss_weights '{"code_swav": 1.0, "distillation": 0.1}' \
  --epochs 5 --lr 5e-5 --warmup_steps 1000 --batch_size 16 \
  --eval_every 1 --eval_subset_size 100 --save_every 1 --seed 42
```

Pass criteria:
- k=80 mIoU > frozen baseline 52.69% by ≥ +0.5pp.
- k=80 PQ > frozen baseline 27.87% by ≥ +0.2pp.
- If mIoU regresses, first check whether prototypes collapsed (all cluster assignments concentrated in <10 clusters) — if so, raise `--swav_entropy_weight` (0.1 → 0.3).

---

## Cross-cutting: LoRA+ (optional add-on for Exp 2 and 3)

If Exp 2 or 3 show no improvement over Exp 1, enable LoRA+ (Hayou 2024): `B` parameters receive 8× higher LR than `A`. Essentially free quality uplift.

- Replace optimizer construction at `train_semantic_adapter.py:278` with parameter groups:
  - A-params (`.lora_A`, `.lora_A_*`): LR = `args.lr`
  - B-params (`.lora_B`, `.lora_B_*`): LR = `args.lr * args.lora_plus_ratio`
  - SwAV prototypes: LR = `args.lr` (normal)
- Add CLI: `--lora_plus` (action=store_true), `--lora_plus_ratio 8.0`.

Do not enable by default — introduce after Exp 2/3 baseline numbers are in.

---

## Rollout order and decision gates

1. **Exp 1** (1 PR). Must pass: no `inf` spikes, mIoU ≥ frozen − 0.2pp. **Gate**: if mIoU regresses > 0.5pp, stop and investigate before proceeding.
2. **Exp 2** (1 PR, depends on Exp 1). Must pass: trainable params ~73k, mIoU ≥ frozen − 0.5pp. **Gate**: if split-QKV forward is buggy, fix before Exp 3.
3. **Exp 3** (1 PR, depends on Exp 2). Must pass: mIoU > frozen + 0.5pp OR PQ > frozen + 0.2pp. **Gate**: if neither, consider LoRA+ add-on or prototype re-initialization.

Do not combine experiments into a single run. The E2 post-mortem explicitly shows that combined changes make regressions unattributable.

---

## Critical files to modify

| File | Exp | Role |
|------|-----|------|
| `mbps_pytorch/models/adapters/lora_layers.py` | 1, 2 | `pissa_init_` helper; `LoRASplitQKV` class; extend `freeze_non_adapter_params` suffixes |
| `mbps_pytorch/models/adapters/dinov2_adapter.py` | 2 | `attention_mode`, `qv_targets`, `adapt_ffn` args in `inject_lora_into_dinov2` |
| `mbps_pytorch/train_semantic_adapter.py` | 1, 2, 3 | CLI flags, regimen fixes, PiSSA hook, SwAV hook, prototype loading, `adapter_config` extension, early-stop on mIoU |
| `mbps_pytorch/generate_semantic_pseudolabels_adapted.py` | 2 | Read new `adapter_config` fields with backward-compatible defaults |
| `mbps_pytorch/eval_cause_k80.py` | 2 | Read new `adapter_config` fields with backward-compatible defaults |
| `mbps_pytorch/losses/code_swav_loss.py` | 3 | New file — Sinkhorn-Knopp + SwAV loss |
| `mbps_pytorch/tests/test_dora_adapter_training.py` | 1, 2 | Five new tests (2 for PiSSA, 3 for split-QKV) |
| `mbps_pytorch/tests/test_code_swav_loss.py` | 3 | New file — three tests for SwAV correctness |
| `scripts/train_dino_adapter_distill.sh` | 1 | Update command to use `--variant lora --pissa_init --epochs 5 --lr 5e-5 --warmup_steps 1000 --eval_every 1` |

## Reusable utilities to leverage (do not rewrite)

| Utility | Path | Use |
|---------|------|-----|
| `LoRALinear` class | `mbps_pytorch/models/adapters/lora_layers.py:24` | Base class for PiSSA hook |
| `wrap_linear_if_match` | `mbps_pytorch/models/adapters/lora_layers.py:250` | Pattern for split-QKV injection |
| `freeze_non_adapter_params` | `mbps_pytorch/models/adapters/lora_layers.py:297` | Extend suffix list only |
| `count_adapter_params` | `mbps_pytorch/models/adapters/lora_layers.py:312` | Reuse for verification prints |
| `sinkhorn_knopp_teacher` pattern | `refs/dinov3/dinov3/loss/dino_clstoken_loss.py:43-70` | Port to single-GPU PyTorch Sinkhorn |
| `transform` code projection | `refs/cause/modules/segment_module.py` (imported at `train_semantic_adapter.py:46`) | Project seg_feat to 90D spatial |
| `Cluster.cluster_probe` | `refs/cause/modules/segment_module.py:109` | Not used — we use k=80 centroids instead |
| k=80 centroids file | `<data_dir>/cityscapes/pseudo_semantic_raw_k80/kmeans_centroids.npz` | Load as SwAV prototype init |
| `evaluate_kmeans` | `mbps_pytorch/eval_cause_k80.py:523-613` | Reuse as inline subset eval hook for `--eval_every` |
| `uniformity_loss` pattern | `mbps_pytorch/losses/consistency_loss.py:19-70` | Numerically stable entropy pattern for SwAV regularizer |

## Verification summary

End-to-end after Exp 3:

```
# 1. Local smoke test (5 min, macOS MPS)
bash scripts/eval_dora_vs_frozen_k80_local.sh  # with ADAPTER_CKPT pointing at new run

# 2. Full comparison run on remote (grep the SUMMARY lines)
bash scripts/eval_dora_vs_frozen_k80.sh
grep -E "^K=" logs/eval_k80_frozen_baseline.log
grep -E "^K=" logs/eval_k80_dora_adapter.log

# 3. Unit tests
pytest mbps_pytorch/tests/test_dora_adapter_training.py -v
pytest mbps_pytorch/tests/test_code_swav_loss.py -v
```

Expected outcome after all three experiments:
- **Exp 1**: parity with frozen baseline (mIoU 52.69% ± 0.2pp), zero `inf` spikes, no init hacks.
- **Exp 2**: parity with Exp 1, trainable params drop ~6× (463k → ~73k).
- **Exp 3**: mIoU > 53.2%, PQ > 28.1% (beating frozen baseline for the first time on this pipeline).
