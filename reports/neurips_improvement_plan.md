# Plan: NeurIPS Paper 4/10 → 9/10

## Context

A simulated NeurIPS review scored our paper 4/10. The review identified 7 major weaknesses: (W1) low novelty, (W2) DINOv3 backbone confound, (W3) cherry-picked metrics, (W4) SPIdepth not truly monocular, (W5) unexplained PQ_things jump, (W6) single seed, (W7) no COCO evaluation. The "What Would Change My Mind" section requires 5 specific experiments. This plan addresses all of them.

**Critical finding:** `cups_pseudo_labels_k80/` = our monocular pseudo-labels (confirmed via `convert_to_cups_format.py` line 36-45). The PQ=32.76 result IS from our monocular labels + DINOv3. The CUPS stereo + DINOv3 control is genuinely missing.

---

## Phase 1: Critical Experiments (Week 1) — Run in Parallel

### 1A. CUPS Stereo + DINOv3 Control [W2] — Remote 2x 1080 Ti
**The #1 experiment. Determines if our paper has a contribution at all.**

1. Generate CUPS stereo pseudo-labels using `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py`
   - Requires: Cityscapes stereo sequences, RAFT optical flow, sf2se3 scene flow
   - Output: `cups_stereo_pseudo_labels/` in flat CUPS format
2. Create config `train_cityscapes_dinov3_vitb_stereo_2gpu.yaml` (clone k80 config, change ROOT_PSEUDO)
3. Train Stage 2 (10K steps) + Stage 3 (8K steps) with DINOv3 ViT-B/16
4. Evaluate on full 500-image val set

**If CUPS_stereo+DINOv3 < 31 PQ:** Our pseudo-labels contribute 1.7+ points. Paper is strong.
**If CUPS_stereo+DINOv3 ~ 31-33:** Marginal. Reframe around monocular applicability.
**If CUPS_stereo+DINOv3 > 34:** Our contribution is near-zero. Paper needs fundamental rethink.

Time: ~3 days. Files: `refs/cups/cups/pseudo_labels/gen_pseudo_labels.py`, `refs/cups/configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml`

### 1B. Depth Model Ablation [W4] — Anydesk A6000
**Proves "monocular" claim by using depth models NOT trained on Cityscapes.**

1. Generate Depth Anything V2 maps: `python mbps_pytorch/generate_depth_maps.py --use_dav2 --data_dir .../leftImg8bit/{train,val} --output_dir .../depth_dav2/{train,val}`
2. Generate Depth Anything V3 maps: same script, no `--use_dav2` flag, output to `depth_dav3/`
3. For each depth model, run instance splitting: `python mbps_pytorch/generate_depth_guided_instances.py --depth_dir .../depth_dav2/... --grad_threshold 0.20 --min_area 1000`
4. Convert to CUPS format: `python unsupervised-panoptic-segmentation/pseudo_labels/convert_to_cups_format.py` with `--depth_subdir depth_dav2`
5. Evaluate pseudo-label PQ for each depth model

**Target:** DA2/DA3 PQ_things within 2 points of SPIdepth's 19.41.
Time: ~8 hours. Files: `mbps_pytorch/generate_depth_maps.py`, `mbps_pytorch/generate_depth_guided_instances.py`

### 1C. Per-Class Stage-3 Trajectory [W5] — Local (no compute needed)
**Explains the PQ_things 23.17→34.13 jump.**

1. Parse all 7 Stage-3 eval logs from `experiments/val_dinov3_official_stage3_step*.log`
2. Extract per-class PQ/SQ/RQ at every checkpoint (steps 600, 800, 1800, 2000, 3400, 5200, 8000)
3. Create trajectory table showing smooth growth (the jump IS smooth per the plan agent's analysis)
4. Identify which thing classes drive the improvement (car? truck? vs person?)

Time: ~2 hours (scripting only).

### 1D. Stuff-Things Classifier Ablation [W9] — Local
**Fills the unablated 4th component gap.**

Three conditions: (1) Oracle GT stuff/things labels, (2) DINOv3 attention classifier, (3) Depth-gradient classifier. For each, regenerate instances and evaluate pseudo-label PQ.

Time: ~3 hours. Files: `mbps_pytorch/classify_stuff_things.py`, `mbps_pytorch/classify_stuff_things_attention.py`

---

## Phase 2: Statistical Rigor + Generalization (Week 2)

### 2A. Multi-Seed Runs [W6] — Split across machines
**3 seeds (existing 1996, plus 42 and 2026). Mean +/- std on all metrics.**

Run 2 additional full pipelines (Stage 2 + Stage 3):
```bash
# Seed 42 (Anydesk A6000)
python train.py --experiment_config_file configs/train_cityscapes_dinov3_vitb_k80_2gpu.yaml SYSTEM.SEED 42
python train_self.py --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_k80_2gpu.yaml SYSTEM.SEED 42

# Seed 2026 (Remote 1080 Ti, after Exp 1A finishes)
python train.py ... SYSTEM.SEED 2026
python train_self.py ... SYSTEM.SEED 2026
```

Time: ~4 days (2 runs, can partially overlap on different machines).

### 2B. COCO-Stuff-27 Evaluation [W7] — Anydesk A6000
**Multi-dataset evaluation for generalizability.**

1. Download COCO-Stuff-27 data, generate CAUSE features on COCO images
2. Overclustering + Depth Anything V2 (domain-agnostic) for COCO pseudo-labels
3. Train Stage 2 + Stage 3 on COCO
4. Evaluate. Use existing config template: `configs/coco_stuff27.yaml`

Time: ~4-5 days. Depends on compute availability after 2A.
Files: `configs/coco_stuff27.yaml`, `mbps_pytorch/generate_coconut_pseudolabels.py`

---

## Phase 3: Novelty + Paper Rewrite (Week 2-3)

### 3A. Novelty Reframing [W1]
The reviewer is right: K-means + Sobel are elementary. Two-pronged strategy:

**Prong 1 — Reframe as empirical contribution:**
- Title: "Monocular Depth Gradients Suffice for Unsupervised Panoptic Segmentation"
- The contribution is the FINDING (monocular suffices, validated by depth ablation), not the TECHNIQUE
- The instance method comparison (8 methods, Table 6) becomes a centerpiece
- The overclustering-depth interaction is an analytical contribution, not a technique contribution

**Prong 2 — Acknowledge CUPS overclustering precedent:**
- Cite CUPS Table 7b (k=54 → PQ=30.6) explicitly
- Frame our overclustering as extending CUPS's finding to the monocular setting with depth interaction analysis
- The novelty is the INTERACTION between overclustering and depth, not overclustering itself

### 3B. Paper Restructure [W8]
Complete rewrite from project report → focused 9-page conference paper:

**New structure:**
1. Abstract (rewritten: lead with finding, not pipeline)
2. Introduction (2pp): Problem → stereo gap → monocular sufficiency → contributions
3. Related Work (0.75pp): Compact
4. Method (2.5pp): Clean pipeline + proper figures (matplotlib/tikz, NOT ASCII)
5. Experiments (3pp): All new tables with CUPS+DINOv3 control, depth ablation, multi-seed, COCO
6. Analysis (0.5pp): Overclustering-depth tradeoff, per-class trajectory
7. Conclusion (0.25pp)
8. **Supplementary**: ALL failed approaches, engineering stories, metric confusion

**Specific fixes:**
- [W3] Stop cherry-picking PQ_things. Lead with overall PQ. Fill Table 1 CUPS PQ_stuff.
- [W11] Complete Table 1 (CUPS PQ_stuff = 35.10 from their paper Table 1)
- [W12] Add U2Seg (PQ=18.4 from `informations/dataset_into.md`) to comparison table
- [W13] Fix table numbering (Tables 8a/8b → Table 8, renumber Table 9)
- [W14] Clarify "7 dead classes" = 14/27 CAUSE centroids never win argmax → 7/19 eval classes get 0 IoU
- [W15] Replace ASCII figures with proper publication figures
- [W16] Fix DINOv3 citation (add arXiv ID or official reference)

### 3C. New Core Table (Paper Centerpiece)

| Method | Backbone | Pseudo-labels | PQ | PQ_th | PQ_st |
|--------|----------|--------------|---:|------:|------:|
| U2Seg | - | ImageNet | 18.4 | - | - |
| CUPS | DINO RN50 | Stereo | 27.8 | 17.7 | 35.1 |
| **CUPS** | **DINOv3 ViT-B** | **Stereo** | **[Exp 1A]** | **??** | **??** |
| Ours | DINO RN50 | Mono (SPIdepth) | 24.7 | - | - |
| Ours | DINOv3 ViT-B | Mono (SPIdepth) | 32.8±X | 34.1±X | 32.0±X |
| Ours | DINOv3 ViT-B | Mono (DA3) | [Exp 1B] | ?? | ?? |

---

## Execution Timeline

```
Week 1:
  Remote 1080 Ti: ████ Exp 1A (CUPS stereo+DINOv3) ████████
  Anydesk A6000:  ██ Exp 1B (depth ablation) ██ Exp 2A seed 42 ████
  Local M4 Pro:   █ 1C (logs) █ 1D (stuff-things) █ Paper draft ████

Week 2:
  Remote 1080 Ti: ████ Exp 2A seed 2026 ████████
  Anydesk A6000:  ████ Exp 2B (COCO) ████████████████
  Local M4 Pro:   ████ Paper rewrite + figures ████████████████

Week 3:
  All machines:   Results compilation, final paper, supplementary, polish
```

---

## Risk Matrix

| Risk | Impact | Mitigation |
|------|--------|------------|
| CUPS_stereo+DINOv3 > 34 PQ | Fatal — no contribution | Reframe as monocular applicability paper |
| DA2/DA3 PQ_things < 14 | Weakens "truly monocular" | Show DA still beats spectral methods (10.05) |
| COCO PQ < 15 | Weakens generalization | Position as driving-domain contribution |
| Multi-seed std > 1.5 | Weakens all claims | Run 5 seeds instead of 3 |
| CUPS stereo pseudo-label code broken | Delays Exp 1A | Use CUPS's published pseudo-labels if available |

---

## Verification

After all experiments complete:
1. All tables have mean±std from 3 seeds
2. CUPS+DINOv3 row proves pseudo-labels matter (or we reframe)
3. DA2/DA3 row proves depth model generality
4. COCO row proves dataset generality
5. Per-class trajectory explains PQ_things growth
6. Stuff-things ablation fills the gap
7. Paper passes self-review against all 16 review weaknesses (W1-W16)
8. All figures are publication-quality (matplotlib/tikz)
9. No ASCII art remains
10. Supplementary contains all cut content (failed approaches, engineering bugs)
