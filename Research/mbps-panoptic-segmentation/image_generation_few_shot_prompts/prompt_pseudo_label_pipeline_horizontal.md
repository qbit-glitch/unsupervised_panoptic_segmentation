## Prompt: Full 4-Step Depth-Conditioned Pseudo-Label Pipeline (Horizontal Layout)

```md
Create a professional ML research paper architectural diagram showing a complete
4-step depth-conditioned pseudo-label generation pipeline for unsupervised
panoptic segmentation on Cityscapes. Clean, minimal, publication-quality style
similar to CVPR/ICCV paper figures. White background, no clutter, dense but
readable. Layout flows horizontally left-to-right.

=== LAYOUT ===
Single wide horizontal flowchart with 6 major columns flowing left-to-right,
separated by thin bold vertical divider lines.
Left-to-right order:
  [1] INPUT COLUMN        — light gray background #FAFAFA
  [2] STEP 1 COLUMN       — blue accent #1565C0
  [3] STEP 2 COLUMN       — green accent #2E7D32
  [4] STEP 3 COLUMN       — orange accent #E65100
  [5] STEP 4 COLUMN       — purple accent #6A1B9A
  [6] OUTPUT COLUMN       — teal accent #00695C
Overall aspect ratio: ultra-wide landscape (roughly 21:9 or 3:1)

=== INPUT COLUMN (far left, gray #FAFAFA background) ===
Header: "INPUT" in dark gray, bold.

Vertical stack of two input boxes:

[1] INPUT BOX (top):
    - Small thumbnail of a Cityscapes RGB street scene
    - Label: "RGB Image"
    - Subtext: "leftImg8bit.png, 1024 × 2048"

[2] INPUT BOX (bottom):
    - Small depth map visualization (viridis/plasma colormap)
    - Label: "DepthPro Monocular Depth"
    - Subtext: "512 × 1024, normalized [0,1]"

A thin vertical arrow labeled "DepthPro inference" connects [1] to [2].
Two outgoing arrows from this column: top arrow (from RGB) goes to STEP 1,
bottom arrow (from Depth) goes to STEP 2.

=== STEP 1 COLUMN — SEMANTIC PSEUDO-LABEL GENERATION (blue #1565C0 accent) ===
Header bar: full-height left edge blue stripe, or top banner with blue
background #1565C0 and white bold text: "STEP 1 — SEMANTIC PSEUDO-LABEL"
Subtext: "Feature-level: DINOv2 ViT-B/14 (CAUSE) → 90D codes"

Vertical top-to-bottom flow inside this column:

[1] FEATURE BOX (top):
    - Icon: neural network 3D slabs (blue gradient)
    - Label: "CAUSE Segment_TR"
    - Subtext: "90-D per patch (32×64)"
    - Lock/snowflake icon (frozen)

[2] PARALLEL (right of [1], connected by horizontal arrow):
    - Wave/sine icon
    - Label: "Sinusoidal Depth Enc"
    - Subtext: "16-D, 8 freq bands"

[3] MERGE ARROW down into:

[4] DCFA BOX (center, prominent, white with blue border):
    - Title: "DCFA Adapter (~40K)"
    - Mini network: "[106] → [384] → [384] → [90]"
    - Badge: "ZERO-INIT" with green check
    - Skip: "codes + residual"
    - Loss: "L_depth-corr + 20.0 × ||A_θ(f,d) − f||²"

[5] ARROW down to:

[6] CLUSTERING BOX:
    - Scatter/cluster icon
    - Label: "MiniBatchKMeans"
    - Subtext: "k=80 | batch=10K | n_init=3"

[7] ARROW down to:

[8] ASSIGN BOX:
    - Label: "Assign & Upsample"
    - Subtext: "32×64 → 512×1024 NN"

[9] ARROW down to:

[10] MAP BOX:
    - Label: "Cluster→Class Map"
    - Subtext: "Val majority vote | 80 → 19 trainIDs"

[11] OUTPUT (bottom of column):
    - Small semantic map thumbnail
    - Label: "SEMANTIC PL"
    - Subtext: "trainIDs 0–18"

An arrow exits the bottom of this column, curves right, and feeds into STEP 4.

=== STEP 2 COLUMN — DEPTH-GUIDED INSTANCE GENERATION (green #2E7D32 accent) ===
Header: "STEP 2 — INSTANCE GENERATION" in white on green #2E7D32.
Subtext: "Instance-level: DepthPro + Sobel + CC | NO vision features"

Vertical top-to-bottom flow:

[1] INPUT (top):
    - Depth map thumbnail (viridis)
    - Label: "Depth Map"

[2] THREE SMALL BOXES side-by-side (branches):
    Left:   "Gaussian σ=0.0"
    Center: "Sobel Edge"
    Right:  "Thing Masks"

[3] ARROWS converge to:

[4] THRESHOLD BOX:
    - Label: "Depth Edge Threshold"
    - Formula: "||∇D|| > τ = 0.20"

[5] ARROW down to:

[6] CC BOX (white with green border):
    - Title: "Per-Class Connected Components"
    - Mini list:
      "M_c = semantic==c"
      "M'_c = M_c \\ edges"
      "CC(M'_c) → filter ≥1000px"
      "Dilate 3px"

[7] OUTPUT (bottom):
    - Instance mask thumbnail (random colors on black)
    - Label: "INSTANCE PL"
    - Subtext: "~17 valid | uint16"

An arrow exits the bottom, curves right to STEP 4.

=== STEP 3 COLUMN — SIMCF-ABC FILTERING (orange #E65100 accent) ===
Header: "STEP 3 — SIMCF-ABC FILTERING" in white on orange #E65100.
Subtext: "Label-level: mutual consistency | PQ +0.73"

Two inputs at top (fed by arrows curving in from STEP 1 and STEP 2 outputs):

[1] INPUT (top-left):
    - Semantic map thumbnail
    - Label: "Semantic PL"

[2] INPUT (top-right):
    - Instance mask thumbnail
    - Label: "Instance PL"

Vertical stack below:

[3] STEP A BOX:
    - Title: "A: Instance → Semantics"
    - Subtext: "Majority vote per instance"
    - Badge: "No-op (0 px changed)"

[4] ARROW down to:

[5] STEP B BOX (larger, ★ critical):
    - Title: "B: Semantics → Instances ★"
    - Subtext: "Adjacency (d=3px) + union-find merge"
    - Result badges: "44→22 inst", "+1.33 PQ_things"
    - Badge: "NO DINOv2/DINOv3"

[6] ARROW down to:

[7] STEP C BOX:
    - Title: "C: Depth → Semantics"
    - Subtext: "3-sigma outlier mask"
    - Formula: "|D(p) − μ_c| > 3σ_c"
    - Badge: "~85M px masked | +0.30 PQ_stuff"

[8] OUTPUT (bottom):
    - Small thumbnail showing refined semantic + instance side-by-side
    - Label: "REFINED LABELS"

Arrows exit bottom to STEP 4.

=== STEP 4 COLUMN — PANOPTIC MERGE (purple #6A1B9A accent) ===
Header: "STEP 4 — PANOPTIC MERGE" in white on purple #6A1B9A.
Subtext: "Encoding: panoptic_id = class_id × 1000 + instance_id"

Three inputs at top (arrows from STEP 1, STEP 2, STEP 3 outputs):

[1] INPUT (left): "Semantic Map"
[2] INPUT (center): "Instance Map"
[3] INPUT (right): "stuff_things.json"

All arrows into:

[4] MERGE BOX (large, white with purple border):
    - Title: "generate_panoptic_map()"
    - Three mini steps with small icons:
      "① Place Things: score-sort, majority cls, overlap check"
      "② Place Stuff: fill unassigned, area≥64, id=0"
      "③ Fallback CC: uncovered → new instances, score=0.1"

[5] OUTPUT (bottom):
    - Panoptic prediction thumbnail (Cityscapes panoptic colors)
    - Label: "PANOPTIC PL"
    - Subtext: ".npy (int32) | .png (uint16) | .json"

Arrow exits right to OUTPUT column.

=== OUTPUT COLUMN (far right, teal #00695C accent) ===
Header: "DOWNSTREAM TRAINING" in white on teal #00695C.

Vertical stack:

[1] OUTPUT BOX (top):
    - Panoptic thumbnail
    - Label: "Panoptic Pseudo-Labels"
    - Badge: "PQ = 25.85"
    - Subtext: "2,975 images × 3 files"

[2] THREE SMALL BOXES side-by-side below:
    Left:   "Mask2Former frozen DINOv3 ~28% PQ"
    Center: "Cascade Mask R-CNN"
    Right:  "Self-Training → 35.83% PQ ★"

[3] BOTTOM NOTE (small, red text):
    "DINOv3 ViT-B/16 = FROZEN training backbone — NOT used in pseudo-label generation"

=== BOTTOM METRICS BAR ===
Thin full-width horizontal bar spanning all columns at the very bottom:
Left:  "DCFA +0.68 PQ  |  SIMCF-ABC +0.73 PQ"
Center: "Compositional +1.31 PQ (24.54 → 25.85)"
Right: "Orthogonal: feature + instance + label levels"

=== STYLE GUIDELINES ===
- White background overall
- Each column has a very subtle tinted background:
  INPUT=#FAFAFA, STEP1=#E3F2FD, STEP2=#E8F5E9, STEP3=#FFF3E0, STEP4=#F3E5F5, OUTPUT=#E0F2F1
- Column headers: colored banner at top of each column, white bold text, full column width
- Processing boxes: white fill, thin colored border matching step accent, rounded corners (radius 4px)
- Arrows: thin black with arrowheads, orthogonal right-angle routing, labeled where needed
- Curved arrows: for feedback/loop paths (e.g., STEP 1/2 outputs curving to STEP 3/4),
  use smooth 90-degree curves with arrowheads
- Real image thumbnails: ~120px wide, consistent size, embedded in input/output boxes
  (Cityscapes RGB, viridis depth, semantic color map, instance random colors, panoptic overlay)
- Formula/equations: clean serif math font, inline, small but legible
- Badges: small rounded pill shapes — green (#4CAF50) for positive results,
  yellow (#FFC107) for notes, red (#F44336) for critical warnings
- Neural network icons: 3D slab-style layers in muted blue gradient
- Lock/snowflake icon next to all frozen models
- Typography: sans-serif throughout (Helvetica or Arial style), strict hierarchy
  — column headers 14pt, box titles 11pt, subtext 9pt, formulas 9pt serif
- Flat design, no drop shadows, no gradients except 3D network slab icon
- Dense but readable — typical of a full-pipeline figure in a top-tier CV paper
- Vertical dividers between columns: thin 2px line in matching accent color

=== REFERENCE ASCII LAYOUT ===

┌─ INPUT ─┐  ┌─ STEP 1: SEMANTIC ────────────────────┐  ┌─ STEP 2: INSTANCE ────────────────┐
│ [RGB]   │  │ [CAUSE 90D]  [Depth Enc 16D]         │  │ [Depth Map]                       │
│ [Depth]─┼─→│          ↓                           │  │    ↓                              │
└─────────┘  │      [DCFA ~40K]                     │  │ [Gauss][Sobel][ThingMasks]        │
             │          ↓                           │  │    ↓                              │
             │      [K-Means k=80]                  │  │ [||∇D||>0.20]                     │
             │          ↓                           │  │    ↓                              │
             │      [Assign+Upsample]               │  │ [Per-Class CC + Filter]           │
             │          ↓                           │  │    ↓                              │
             │      [Cluster→Class Map]             │  │ [INSTANCE PL]────────────────┐    │
             │          ↓                           │  └──────────────────────────────┼────┘
             │      [SEMANTIC PL]───────────────┐   │                                 │
             └──────────────────────────────────┼───┘                                 │
                                                │                                     │
                                                └─────────────┬───────────────────────┘
                                                              │
                                                              ▼
             ┌─ STEP 3: SIMCF-ABC ───────────────────────────┐
             │ [Semantic PL] [Instance PL]                   │
             │        ↓                                      │
             │    [A: Majority Vote]                         │
             │        ↓                                      │
             │    [B: Merge ★] (adj+union-find)              │
             │        ↓                                      │
             │    [C: 3-Sigma Mask]                          │
             │        ↓                                      │
             │    [REFINED LABELS]───────────────────────┐   │
             └───────────────────────────────────────────┼───┘
                                                         │
                                                         ▼
             ┌─ STEP 4: PANOPTIC MERGE ──────────────────────┐
             │ [Semantic] [Instance] [stuff_things.json]     │
             │              ↓                                  │
             │      [generate_panoptic_map()]                │
             │         ① Place Things                        │
             │         ② Place Stuff                         │
             │         ③ Fallback CC                         │
             │              ↓                                  │
             │      [PANOPTIC PL]────────────────────────┐   │
             └───────────────────────────────────────────┼───┘
                                                         │
                                                         ▼
             ┌─ OUTPUT: DOWNSTREAM ──────────────────────────┐
             │ [Panoptic PL | PQ=25.85]                      │
             │    ↓                                            │
             │ [Mask2Former] [Cascade R-CNN] [Self-Train 35.83│
             │  ~28% PQ                                      │
             └───────────────────────────────────────────────┘

═══════════════════════════════════════════════════════════════════════════════════════
METRICS: DCFA +0.68 | SIMCF-ABC +0.73 | Total +1.31 PQ (24.54 → 25.85)
═══════════════════════════════════════════════════════════════════════════════════════
```
