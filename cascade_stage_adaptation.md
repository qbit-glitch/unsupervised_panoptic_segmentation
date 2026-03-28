CUPS Dissection: Where the PQ Points Come From                                                                                                                           
   
  ┌────────────────────────┬───────────────────────────────┬──────────────────────────────┬──────────────────────────────────────────────────────────────────────────┐     
  │       Component        │        CUPS's Approach        │            Result            │                                 Weakness                                 │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Backbone               │ DINO ResNet-50                │ mIoU 26.8                    │ Weak features; DINOv1 era                                                │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Instance pseudo-labels │ SF2SE3 on stereo scene flow   │ Sparse (moving objects only) │ Misses static things entirely                                            │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Semantic pseudo-labels │ DepthG on DINO features       │ 26.8 mIoU                    │ Low-res (320x320), needs depth-upsampling hack                           │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Stuff/things split     │ Frequency ratio (ψ=0.08)      │ Decent                       │ Simple threshold, no feature reasoning                                   │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Training network       │ Panoptic Cascade Mask R-CNN   │ Detection-based              │ Overkill for unsupervised; detection head trained on sparse pseudo-masks │
  ├────────────────────────┼───────────────────────────────┼──────────────────────────────┼──────────────────────────────────────────────────────────────────────────┤
  │ Self-training          │ EMA + copy-paste + confidence │ +3.7 PQ                      │ Sound approach                                                           │
  └────────────────────────┴───────────────────────────────┴──────────────────────────────┴──────────────────────────────────────────────────────────────────────────┘

  CUPS numbers: PQ=27.8, SQ=57.4, RQ=35.2, PQ^Th=17.7, PQ^St=35.1

  The critical bottleneck is RQ=35.2 (recognition quality) — CUPS fails to find many segments. And PQ^Th=17.7 is especially weak because scene flow only discovers moving
  things.

  Cascaded Strategy to Beat CUPS

  The "cascade" principle: each stage's improvement compounds downstream. CUPS has 3 stages — we upgrade each with strictly stronger components, and add a 4th stage (our
  novel bridge fusion).

  CUPS Pipeline:                    Our Cascade:

  Stereo video → scene flow         DINOv3 features → MaskCut (no stereo!)
  DINO → DepthG (320×320)          DINOv3 → K-means (full resolution)
  Frequency ratio → stuff/things    Same (it works)
  Cascade Mask R-CNN                Semantic head + Instance head + Mamba Bridge
  EMA self-training                 EMA self-training (same, it works)

  Stage-by-stage PQ gain analysis:

  Stage 1: Feature backbone upgrade (DINOv3 vs DINO v1)

  This is the single biggest lever. CUPS uses DINO (2021) features. DINOv3 (2025) has:
  - Linear probe mIoU: ~81 vs ~30 on Cityscapes
  - Patch features are 2-3x more discriminative for clustering
  - Every downstream component benefits

  Expected gain: This alone should push pseudo-label semantic mIoU from ~27 to ~50+, which cascades into better stuff/things classification, better instance-semantic
  alignment, everything.

  Stage 2: Instance discovery without stereo (MaskCut vs SF2SE3)

  CUPS's biggest limitation: SF2SE3 needs stereo video and only finds moving objects. Static things (parked cars, benches) are invisible.

  MaskCut on DINOv3 features:
  - Works on single images (no stereo needed)
  - Discovers static AND moving objects
  - DINOv3's attention maps are far better for NCut bipartition
  - Expected AR@100 boost: CUPS discovers ~3-5 instances/image (only movers), MaskCut on DINOv3 should find ~10-15

  Expected gain: PQ^Th should jump significantly since we're finding objects CUPS completely misses.

  Stage 3: Feature-level fusion via Mamba Bridge (NOVEL — CUPS has nothing here)

  This is the critical cascade stage CUPS lacks entirely. CUPS fuses semantic + instance only at the label level (majority voting in pseudo-labels). Our Mamba bridge fuses
   at the feature level:

  CUPS fusion:  semantic_labels ∪ instance_masks → majority_vote → panoptic_label
  Our fusion:   semantic_features ⊗ instance_features → Mamba2 SSD → refined_both

  The bridge allows:
  - Instance boundaries to sharpen semantic predictions
  - Semantic context to disambiguate overlapping instances
  - Depth conditioning to help both via FiLM modulation

  Stage 4: Same self-training (CUPS's approach works)

  Keep EMA + copy-paste + confidence thresholding. No need to reinvent this — CUPS showed +3.7 PQ from self-training, and that gain should be similar or larger with better
   starting predictions.

  Concrete Cascaded Pipeline

  ┌─────────────────────────────────────────────────────────┐
  │  OFFLINE (PyTorch, GPU) — Better pseudo-labels          │
  │                                                         │
  │  Image ──→ DINOv3 ViT-B ──→ patch features (768-dim)    │
  │                │                                        │
  │                ├──→ K-means(K=19) + CRF ──→ semantic    │
  │                │         (replaces DepthG)              │
  │                │                                        │
  │                ├──→ MaskCut (iterative NCut) ──→ inst.  │
  │                │         (replaces SF2SE3)              │
  │                │                                        │
  │                └──→ Depth Anything V3 ──→ depth maps    │
  │                          (replaces SMURF stereo)        │
  │                                                         │
  │  Stuff/things ← frequency ratio (same as CUPS, works)   │
  │                                                         │
  │  ──→ TFRecords to GCS                                   │
  ├─────────────────────────────────────────────────────────┤
  │  TRAINING (JAX, TPU) — Feature-level fusion             │
  │                                                         │
  │  Image ──→ DINOv3 ViT-B (frozen) ──→ F (B,N,768)        │
  │                │                                        │
  │                ├──→ Semantic Head ──→ logits (B,N,19)   │
  │                │    (CE vs pseudo-semantic)             │
  │                │                                        │
  │                ├──→ Instance Head ──→ embeds (B,N,64)   │
  │                │    (discriminative loss vs pseudo-inst)│
  │                │                                        │
  │                └──→ Mamba Bridge (NOVEL CASCADE STAGE)  │
  │                     logits + embeds → APB → Depth FiLM  │
  │                     → BiCMS Mamba2 → refined outputs    │
  │                     (CKA + reconstruction loss)         │
  │                                                         │
  │  Phase 1 (epochs 1-5): heads only                       │
  │  Phase 2 (epochs 6-25): + bridge (gate warmup)          │
  │  Phase 3 (epochs 26-40): + self-training (EMA, 3 rds)   │
  ├─────────────────────────────────────────────────────────┤
  │  INFERENCE                                              │
  │                                                         │
  │  Refined logits → argmax → semantic map                 │
  │  Refined embeds → cosine clustering → instance map      │
  │  stuff/things map → panoptic merge                      │
  └─────────────────────────────────────────────────────────┘

  Why Each Cascade Stage Beats CUPS

  ┌─────────────────────┬───────────────────────────────┬─────────────────────────────────────────────────┬────────────────┐
  │      Our Stage      │         Replaces CUPS         │               Why Strictly Better               │ Estimated Δ PQ │
  ├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────┼────────────────┤
  │ DINOv3 backbone     │ DINO v1                       │ mIoU 81 vs 30 — features 3x more informative    │ +5-8           │
  ├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────┼────────────────┤
  │ K-means on DINOv3   │ DepthG at 320×320             │ Full-res clustering; no downscale hack needed   │ +3-5           │
  ├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────┼────────────────┤
  │ MaskCut on DINOv3   │ SF2SE3 stereo flow            │ Finds static + moving things; no stereo needed  │ +3-5           │
  ├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────┼────────────────┤
  │ Mamba Bridge fusion │ No fusion (label voting only) │ Feature-level cross-modal fusion; depth FiLM    │ +2-4           │
  ├─────────────────────┼───────────────────────────────┼─────────────────────────────────────────────────┼────────────────┤
  │ Same self-training  │ Same                          │ Baseline is better → self-training gains larger │ +0-2           │
  └─────────────────────┴───────────────────────────────┴─────────────────────────────────────────────────┴────────────────┘

  Conservative estimate: 27.8 + 8-12 = ~36-40 PQ (well past CUPS)

  What We DON'T Need from CUPS

  1. Stereo video — replaced by MaskCut (monocular)
  2. SMURF optical flow — not needed at all
  3. SF2SE3 motion segmentation — replaced by NCut
  4. Cascade Mask R-CNN — replaced by per-pixel heads + bridge
  5. Low-res depth-upsampling hack — DINOv3 works at full 512×1024

  What We KEEP from CUPS (it works)

  1. Stuff/things classification via frequency ratio — simple and effective
  2. Self-enhanced copy-paste augmentation — could add to our Stage 3
  3. EMA self-training with confidence thresholding — already in our pipeline
  4. Hungarian matching for evaluation — standard protocol

  The Real Question: What's Already Built?

  Looking at your codebase, everything for this cascade is already implemented:

  ┌───────────────────────────────┬────────┬────────────────────────────────────────────────┐
  │       Cascade Component       │ Status │                      File                      │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ DINOv3 feature extraction     │ Done   │ mbps_pytorch/extract_dinov3_features.py        │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ K-means semantic              │ Done   │ mbps_pytorch/generate_semantic_pseudolabels.py │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ MaskCut instance              │ Done   │ mbps_pytorch/generate_instance_pseudolabels.py │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Depth Anything V3             │ Done   │ mbps_pytorch/generate_depth_maps.py            │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Stuff/things classification   │ Done   │ mbps_pytorch/classify_stuff_things.py          │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ TFRecord generation           │ Done   │ mbps_pytorch/generate_v2_tfrecords.py          │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ DINOv3 JAX backbone           │ Done   │ mbps/models/backbone/dinov3_vitb.py            │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Semantic + Instance heads     │ Done   │ mbps/models/mbps_v2_model.py                   │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Mamba Bridge (APB+UDCM+BiCMS) │ Done   │ mbps/models/bridge/                            │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Training script               │ Done   │ scripts/train_v2.py                            │
  ├───────────────────────────────┼────────┼────────────────────────────────────────────────┤
  │ Self-training (EMA)           │ Done   │ Built into trainer                             │
  └───────────────────────────────┴────────┴────────────────────────────────────────────────┘

  The only missing piece is self-enhanced copy-paste augmentation (CUPS's contribution that helps bootstrapping) — we could add this as a data augmentation in the training
   pipeline if needed.

  Bottom Line

  CUPS's main weakness is: weak features (DINO v1) + stereo dependency (SF2SE3) + no feature-level fusion. Our cascade replaces all three bottlenecks with strictly
  stronger components. The Mamba bridge is the novel cascade stage that CUPS has no equivalent for — it's the only component in this space doing feature-level cross-modal
  fusion rather than label-level fusion.
