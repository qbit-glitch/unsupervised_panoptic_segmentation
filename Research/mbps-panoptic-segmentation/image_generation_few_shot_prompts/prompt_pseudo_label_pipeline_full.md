## Prompt: Full 4-Stage Depth-Conditioned Pseudo-Label Pipeline

```md
Create a professional ML research paper architectural diagram showing a complete
4-stage depth-conditioned pseudo-label generation pipeline for unsupervised
panoptic segmentation on Cityscapes. Clean, minimal, publication-quality style
similar to CVPR/ICCV paper figures. White background, no clutter, dense but
readable.

=== LAYOUT ===
Single tall vertical flowchart with 6 major horizontal bands stacked top-to-bottom,
separated by thin bold divider lines.
Top-to-bottom order:
  [1] INPUT BAND        — light gray background #FAFAFA
  [2] STAGE 1 BAND      — blue accent #1565C0
  [3] STAGE 2 BAND      — green accent #2E7D32
  [4] STAGE 3 BAND      — orange accent #E65100
  [5] STAGE 4 BAND      — purple accent #6A1B9A
  [6] DOWNSTREAM BAND   — teal accent #00695C
Overall aspect ratio: tall portrait (roughly 2:3 or 9:16)

=== INPUT BAND (top, gray #FAFAFA background) ===
Centered title: "INPUT: Cityscapes Training Set (2,975 images)" in dark gray.

Left-to-right flow:
[1] INPUT BOX (left):
    - Small thumbnail of a Cityscapes RGB street scene (1024×2048)
    - Label: "RGB Image (leftImg8bit.png)"
    - Small text: "1024 × 2048"

[2] ARROW → labeled "DepthPro inference"

[3] INPUT BOX (right):
    - Small depth map visualization (viridis/plasma colormap, 512×1024)
    - Label: "DepthPro Monocular Depth"
    - Small text: "512 × 1024, normalized [0,1]"

Both boxes have thin borders. Subtle downward arrows from both boxes lead into
Stage 1 and Stage 2 respectively.

=== STAGE 1 BAND — SEMANTIC PSEUDO-LABEL GENERATION (blue #1565C0 accent) ===
Header bar: full-width, blue background #1565C0, white bold text:
"STAGE 1 — FEATURE LEVEL: SEMANTIC PSEUDO-LABEL GENERATION"
Subtext (smaller, below): "Backbone: DINOv2 ViT-B/14 (CAUSE) → 90D Segment_TR codes"

Left-to-right flow, then bottom loop:

[1] FEATURE EXTRACTION BOX (left):
    - Icon: neural network layers (3D rectangular slabs, blue gradient)
    - Label: "CAUSE Segment_TR Codes"
    - Subtext: "90-D per patch (32×64) from DINOv2 ViT-B/14"
    - Small lock/snowflake icon (frozen backbone)

[2] PARALLEL BOX (right of [1], same height):
    - Wave/sine curve icon
    - Label: "Sinusoidal Depth Encoding"
    - Subtext: "8 freq bands × [sin, cos] = 16-D"
    - Smaller text: "freqs: {1,2,4,8,16,32,64,128} × πd"

[3] MERGE ARROW down from both [1] and [2] into:

[4] DCFA ADAPTER BOX (center, prominent, white rounded rectangle with blue border):
    - Title: "DCFA: Depth-Conditioned Feature Adapter (~40K params)"
    - Inside, a mini network diagram:
        "[106-D] → Linear(106→384) + LN + ReLU → Linear(384→384) + LN + ReLU → Linear(384→90)"
    - Small badge: "ZERO-INITIALIZED" (green checkmark)
    - Skip arrow labeled: "adjusted_codes = codes + residual"
    - Loss formula below: "L_depth-corr + λ_preserve × ||A_θ(f,d) − f||²  (λ = 20.0)"

[5] ARROW down to:

[6] CLUSTERING BOX:
    - Scatter/cluster icon (points grouped into 80 clusters)
    - Label: "MiniBatchKMeans (k = 80)"
    - Subtext: "batch_size=10K | max_iter=300 | n_init=3 | random_state=42"
    - Smaller text: "80 centroids in 90-D space"

[7] ARROW down to:

[8] ASSIGNMENT BOX:
    - Label: "Cluster Assignment & Upsampling"
    - Subtext: "32×64 patches → nearest-neighbor → 512×1024"

[9] ARROW down to:

[10] MAPPING BOX:
    - Label: "Cluster-to-Class Mapping (Val Majority Vote)"
    - Formula: "class(c) = argmax_t count(cluster=c, GT=t)"
    - Subtext: "80 clusters → 19 Cityscapes trainIDs"

[11] OUTPUT BOX (bottom of Stage 1 band):
    - Small colorful semantic segmentation map thumbnail
    - Label: "SEMANTIC PSEUDO-LABEL"
    - Subtext: "trainIDs 0–18, 512×1024"

=== STAGE 2 BAND — INSTANCE GENERATION (green #2E7D32 accent) ===
Header bar: full-width, green background #2E7D32, white bold text:
"STAGE 2 — INSTANCE LEVEL: DEPTH-GUIDED INSTANCE GENERATION"
Subtext: "Input: ONLY DepthPro depth + semantic thing masks | NO vision features"

Top-down branching flow:

[1] INPUT BOX (top-center):
    - Depth map thumbnail (same viridis colormap)
    - Label: "DepthPro Depth Map (512×1024)"

[2] THREE PARALLEL BRANCHES down from [1]:
    Left branch:   "Gaussian Smoothing (σ = 0.0)"
    Center branch: "Sobel Edge Detection (Gx, Gy)"
    Right branch:  "Class Mask Extraction (thing classes)"
    All three as small white rounded rectangles.

[3] ARROWS converge to:

[4] THRESHOLD BOX:
    - Label: "Depth Edge Threshold"
    - Formula: "||∇D|| > τ = 0.20"

[5] ARROW down to:

[6] CC EXTRACTION BOX (large, white with green border):
    - Title: "Per-Class Connected Component Extraction"
    - Numbered list inside (small text):
      "1. M_c = {pixels where semantic == c}"
      "2. M'_c = M_c \\ depth_edges"
      "3. {CC₁, CC₂, ...} = connected_components(M'_c)"
      "4. Filter: area ≥ A_min = 1000 px"
      "5. Dilate 3 iterations → reclaim boundaries"

[7] OUTPUT BOX (bottom):
    - Colorful instance mask thumbnail (random bright colors per instance on black bg)
    - Label: "INSTANCE PSEUDO-LABELS"
    - Subtext: "~17 valid instances/image | uint16 instance IDs"

=== STAGE 3 BAND — SIMCF-ABC FILTERING (orange #E65100 accent) ===
Header bar: full-width, orange background #E65100, white bold text:
"STAGE 3 — LABEL LEVEL: SEMANTIC-INSTANCE MUTUAL CONSISTENCY FILTERING (SIMCF-ABC)"
Subtext: "PQ: 24.54 → 25.27 (SIMCF) | Full DCFA+SIMCF: 24.54 → 25.85 (+1.31 PQ)"

Two inputs at top, three sequential steps below:

[1] INPUT BOX (left):
    - Semantic map thumbnail
    - Label: "Semantic Pseudo-Label (cluster IDs 0–79)"

[2] INPUT BOX (right):
    - Instance mask thumbnail
    - Label: "Instance Pseudo-Label (uint16 IDs)"

Both arrows down into a vertical stack of 3 processing boxes:

[3] STEP A BOX (white with orange border):
    - Title: "STEP A: Instance Validates Semantics (Majority Vote)"
    - Subtext: "Within each instance I_k, majority-vote trainID t*"
    - Badge: "Structurally no-op for CUPS (0 pixels changed)"

[4] ARROW down to STEP B BOX (white with orange border, slightly larger):
    - Title: "STEP B: Semantics Validate Instances (Instance Merging) ★ MOST CRITICAL"
    - Subtext: "Adjacency graph via d=3px dilation → union-find merge → renumber"
    - Result badges (small green boxes): "44→22 inst/img", "5,502→14,965 px median", "PQ_things +1.33"
    - Important badge: "NO DINOv2/DINOv3 features used"

[5] ARROW down to STEP C BOX (white with orange border):
    - Title: "STEP C: Depth Validates Semantics (3-Sigma Outlier Masking)"
    - Subtext: "Pass 1: global μ_c, σ_c | Pass 2: mask if |D(p) − μ_c| > 3σ_c"
    - Result badges: "~85M pixels masked (1.36%)", "PQ_stuff +0.30"

=== STAGE 4 BAND — PANOPTIC MERGE (purple #6A1B9A accent) ===
Header bar: full-width, purple background #6A1B9A, white bold text:
"STAGE 4 — PANOPTIC MERGE: SEMANTIC + INSTANCE → PANOPTIC PSEUDO-LABEL"
Subtext: "Encoding: panoptic_id = class_id × 1000 + instance_id"

Three inputs converge:

[1] INPUT (left): "Refined Semantic Map (trainIDs 0–18 + 255)"
[2] INPUT (center): "Refined Instance Map (uint16 IDs)"
[3] INPUT (right): "stuff_things.json (unsupervised split)"

All arrows into:

[4] MERGE ALGORITHM BOX (large, white with purple border):
    - Title: "generate_panoptic_map()"
    - Three numbered steps with small icons:
      "Step 1 — PLACE THINGS: sort by score, majority class, mask overlap check"
      "Step 2 — PLACE STUFF: fill unassigned, area ≥ 64 px, instance_id = 0"
      "Step 3 — FALLBACK CC: uncovered thing pixels → new CC instances, score=0.1"

[5] OUTPUT BOX (bottom):
    - Panoptic prediction thumbnail (Cityscapes panoptic colors)
    - Label: "PANOPTIC PSEUDO-LABELS"
    - Subtext: "_panoptic.npy (int32) | _panoptic.png (uint16) | segment_info (JSON)"
    - Badge: "2,975 images × 3 files = ~8,925 outputs"

=== DOWNSTREAM BAND (teal #00695C accent) ===
Header bar: full-width, teal background #00695C, white bold text:
"DOWNSTREAM: SUPERVISED TRAINING"
Subtext (smaller, red/important): "NOTE: DINOv3 ViT-B/16 is the FROZEN training backbone — NOT used in pseudo-label generation"

Left-to-right three boxes:

[1] "Stage 2: Mask2Former (frozen DINOv3) ~28% PQ"
[2] "Stage 2: Cascade Mask R-CNN"
[3] "Stage 3: Self-Training → Final: 35.83% PQ" with upward arrow and star

All three as teal-bordered white boxes with small model icons.

=== BOTTOM METRICS BAR ===
Full-width thin bar at very bottom:
Left:  "DCFA +0.68 PQ  |  SIMCF-ABC +0.73 PQ"
Right: "Compositional: +1.31 PQ total (24.54 → 25.85)"
Center small text: "Orthogonal interventions at feature, instance, and label levels"

=== STYLE GUIDELINES ===
- White background overall, each stage band has a very subtle tinted background
  (blue #E3F2FD, green #E8F5E9, orange #FFF3E0, purple #F3E5F5, teal #E0F2F1)
- Header bars: solid bold color (#1565C0, #2E7D32, #E65100, #6A1B9A, #00695C)
  with white text, full width of the band
- Processing boxes: white fill, thin colored border matching stage accent, rounded corners
- Arrows: thin black with arrowheads, orthogonal (right-angle turns), labeled where needed
- Real image thumbnails: Cityscapes RGB, viridis depth map, semantic color map,
  instance random-color map, panoptic overlay — all at small uniform size (~150px wide)
- Formula/equations: rendered in clean serif math font, inline where space permits
- Badges: small rounded pills — green for positive results, yellow for warnings/notes,
  red for critical notes
- Neural network icons: 3D slab-style layers in blue gradient for DINOv2/CAUSE
- Lock/snowflake icon next to all "frozen" models
- Typography: sans-serif for labels, consistent hierarchy (header > title > subtext > formula)
- Flat design, no drop shadows, no gradients except the 3D network slab icon
- Dense but readable — typical of a full-pipeline figure in a top-tier CV paper

=== REFERENCE ASCII LAYOUT ===

┌─ INPUT ──────────────────────────────────────────────────────────────────────────────┐
│  [RGB 1024×2048] ──→ [DepthPro 512×1024]                                            │
└────────┬─────────────────────────────┬───────────────────────────────────────────────┘
         │                             │
         ▼                             ▼
┌─ STAGE 1: SEMANTIC PSEUDO-LABELS (blue) ─────────────────────────────────────────────┐
│  [CAUSE 90-D] ──┐                                                                   │
│  [Depth Enc 16-D]├─→ [DCFA Adapter] ──→ [K-Means k=80] ──→ [Assign+Upsample]        │
│                 │              (~40K params)           (32×64 → 512×1024)           │
│                 └──────────────────────────────────────────────────────────────────   │
│                                        ↓                                             │
│                           [Cluster→Class Mapping (Val Vote)]                         │
│                                        ↓                                             │
│                           [SEMANTIC PSEUDO-LABEL (trainIDs)]                         │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ STAGE 2: INSTANCE GENERATION (green) ───────────────────────────────────────────────┐
│  [Depth Map] ──→ [Gaussian] [Sobel Edge] [Class Masks] ──→ [||∇D|| > 0.20]         │
│                                                              ↓                       │
│              [Per-Class CC: split → CC → filter ≥1000px → dilate 3]                  │
│                                                              ↓                       │
│              [INSTANCE PSEUDO-LABELS (~17 valid, uint16)]                            │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │                             │
         ▼                             ▼
┌─ STAGE 3: SIMCF-ABC FILTERING (orange) ──────────────────────────────────────────────┐
│  [Semantic PL] ──┐                                                                   │
│  [Instance PL] ──┼─→ [Step A: Majority Vote] ──→ [Step B: Merge via Adj+UnionFind]  │
│                  │         (no-op)                      ★ CRITICAL                   │
│                  └────────────────────────────────────────────────────────────────    │
│                                        ↓                                             │
│                           [Step C: 3-Sigma Depth Outlier Mask]                       │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │                             │
         ▼                             ▼
┌─ STAGE 4: PANOPTIC MERGE (purple) ───────────────────────────────────────────────────┐
│  [Refined Semantic] [Refined Instance] [stuff_things.json]                           │
│                       ↓                                                              │
│              [generate_panoptic_map()]                                               │
│                 1. Place Things (score-sort, majority, overlap-check)                │
│                 2. Place Stuff (fill unassigned, area≥64, id=0)                      │
│                 3. Fallback CC (uncovered thing → new instances)                     │
│                       ↓                                                              │
│              [PANOPTIC PSEUDO-LABELS]                                                │
└──────────────────────────────────────────────────────────────────────────────────────┘
         │
         ▼
┌─ DOWNSTREAM: SUPERVISED TRAINING (teal) ─────────────────────────────────────────────┐
│  [Mask2Former] [Cascade Mask R-CNN] [Self-Training → 35.83% PQ]                     │
│       ~28% PQ                                                              ★         │
└──────────────────────────────────────────────────────────────────────────────────────┘
```
