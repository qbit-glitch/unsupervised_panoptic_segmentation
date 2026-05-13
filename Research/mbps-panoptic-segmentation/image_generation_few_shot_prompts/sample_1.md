## Prompt-1

```md
Create a professional ML research paper architectural diagram showing a two-stage
panoptic segmentation training pipeline. Clean, minimal, publication-quality style
similar to CVPR/NeurIPS paper figures. White background, no clutter.

=== LAYOUT ===
Two horizontal sections stacked vertically, separated by a bold dashed line.
Top section: "STAGE-2" (blue accent color #1565C0)
Bottom section: "STAGE-3" (red accent color #C62828)
Overall aspect ratio: wide (roughly 16:9)

=== STAGE-2 (TOP SECTION) ===
Left to right flow:

[1] INPUT BOX (left edge):
    - Small stacked icons representing: "k=80 clusters", "depth maps",
      "stuff/things split ratio"
    - Label: "Pseudo-Labels (from disk)"
    - Include a small real Cityscapes street scene thumbnail

[2] LONG ARROW → pointing right into:

[3] NEURAL NETWORK VISUALIZATION (center):
    - Show as a series of colorful vertical rectangular slabs (like a 3D
      block diagram of network layers), progressively getting narrower
      then wider (encoder-decoder shape)
    - Colors: gradient from blue to green to orange to red across the layers
    - Label below: "Cascade Mask R-CNN + SemSeg Head"
    - Small label above certain layers: "frozen backbone" (with a snowflake
      icon or lock icon) for the left portion, "trainable heads" for the
      right portion
    - Show a small "DropLoss" badge near the detection heads

[4] ARROW → pointing right to:

[5] OUTPUT (right edge):
    - A real Cityscapes panoptic prediction image (colorful segmentation
      overlay on a street scene)
    - Bold label: "PQ = 27.87%"
    - Small config box nearby: "lr=1e-4, SyncBN, 8K steps"

=== STAGE-3 (BOTTOM SECTION) ===
Circular/loop flow:

[1] TEACHER MODEL (top-center of this section):
    - Same colorful layer-slab visualization as Stage-2 but with a
      red/pink tint
    - Label: "TEACHER (EMA, FROZEN)"
    - Snowflake/lock icon indicating frozen

[2] TTA VISUALIZATION (right of teacher):
    - Three copies of the same image at different sizes (small, medium,
      large) representing scales 0.5, 0.75, 1.0
    - Arrows from each scale converging to a single point
    - Label: "TTA (3 scales + flip)"

[3] ARROW down to:

[4] "ONLINE PL FILTERING" box:
    - Clean rounded rectangle
    - Text: "score > 0.5, semantic threshold"
    - Arrow labeled "CopyPaste + PhotometricAug" going right

[5] ARROW down to:

[6] TRAINING LOOP box (center):
    - Shows loss symbol (L in calligraphic font)
    - Label: "bs=16, 12,000 steps"

[7] ARROW left to:

[8] STUDENT MODEL (bottom-left):
    - Same layer-slab visualization with blue tint
    - Label: "STUDENT (head params only)"

[9] CURVED ARROW going UP from student back to teacher:
    - Bold green curved arrow
    - Label on the arrow: "EMA WEIGHT UPDATE"
    - "θ_t = 0.999·θ_t + 0.001·θ_s"
    - This creates a visible circular loop

[10] OUTPUT (bottom-right):
    - A real Cityscapes panoptic prediction image showing improved
      segmentation quality compared to Stage-2
    - Bold label: "PQ = 30.26%"

=== BOTTOM BAR ===
A summary configuration box spanning the full width:
- Left portion: "Stage-2 vs Stage-3" comparison showing:
  "SyncBN → FrozenBN", "DropLoss ON → OFF", "grad clip 0.1 → 1.0"
- Right portion: "Combined Results" showing:
  "Stage-2: PQ=27.87%"
  "Stage-3: PQ=30.26% (step 1800, +2.5)"
  "PQ_things=28.50% vs CUPS 17.7%"

=== STYLE GUIDELINES ===
- Clean white background
- Thin black arrows with arrowheads
- Rounded rectangle boxes with thin borders
- Neural network layers shown as colorful 3D rectangular slabs
  (perspective view, like looking at a bookshelf from an angle)
- Real image thumbnails embedded at input/output stages
  (Cityscapes urban driving scenes with colorful panoptic overlays)
- Minimal text — use icons/visuals over words where possible
- Professional typography (sans-serif, consistent sizing)
- Color scheme: blue (#1565C0) for Stage-2 elements,
  red (#C62828) for Stage-3 elements, green for EMA loop arrow
- No gradients or drop shadows — flat design
- Similar style to CVPR 2025 paper figures

=== REFERENCE ASCII LAYOUT ===

┌─ STAGE-2: TRAIN ON PSEUDO-LABELS ──────────────────────────────────────┐
│                                                                         │
│  [Pseudo-Labels]  ──→  [████████ Network Layers ████████]  ──→  [PQ=27.87%]  │
│   k=80, depth         frozen│trainable    DropLoss                      │
│   split ratio         backbone│heads                                    │
│                                                                         │
└─────────────────────────────────┬───────────────────────────────────────┘
                                  │ checkpoint
┌─ STAGE-3: EMA SELF-TRAINING ────┼──────────────────────────────────────┐
│                                  ▼                                      │
│         ┌──────────────── TEACHER (EMA) ──────────┐                    │
│         │  [████ Frozen Layers ████]               │                    │
│         │         │                                │                    │
│         │         ▼  TTA                           │                    │
│         │    [0.5x] [0.75x] [1.0x] + flip         │                    │
│         │              │                           │                    │
│         │              ▼                           │                    │
│         │     Online PL Filtering                  │                    │
│   EMA   │     + CopyPaste + Aug                    │                    │
│  UPDATE │              │                           │                    │
│    ▲    │              ▼                           │                    │
│    │    │       [Training Loop: L]                 │  ──→  [PQ=30.26%] │
│    │    │              │                           │                    │
│    │    │              ▼                           │                    │
│    │    └──── STUDENT (heads only) ────────────────┘                    │
│    │                   │                                                │
│    └───────────────────┘                                                │
│     θ_t = 0.999·θ_t + 0.001·θ_s                                       │
└─────────────────────────────────────────────────────────────────────────┘
```


## Prompt-2

```md
Create a professional ML research paper architectural diagram showing an
unsupervised panoptic pseudo-label generation pipeline. Clean, minimal,
publication-quality style similar to CVPR/NeurIPS paper figures.

=== LAYOUT ===
Three sections with colored backgrounds:
- Section 1a (top row, light gray #ECECEC background): "Instance Pseudo Labeling"
- Section 1b (bottom row, light gray #ECECEC background): "Semantic Pseudo Labeling"  
- Section 1c (right column, dark gray #666666 background): "Panoptic Assembly"
Section headers use yellow/gold (#FFD54F) highlight badges with bold black text.
Overall aspect ratio: wide landscape (roughly 3:1)

=== SECTION 1a: INSTANCE PSEUDO LABELING (top row, left to right) ===

[1] INPUT: A real Cityscapes urban driving scene (street with cars, pedestrians,
    buildings). Show as a photograph with thin gray border. Label: "RGB Image"

[2] ARROW → into:

[3] PROCESSING BOX: White rounded rectangle with thin black border.
    Text: "Depth Anything v3 (frozen)" with a lock/snowflake icon.
    This is a frozen foundation model.

[4] ARROW → into:

[5] DEPTH MAP IMAGE: A colorful depth map visualization (viridis/plasma colormap)
    of the same Cityscapes scene — nearby objects bright/warm, far objects
    dark/cool. Shows clear depth discontinuities at car boundaries.
    Label: "Depth Map"

[6] ARROW → into:

[7] PROCESSING BOX: White rounded rectangle.
    Text: "Sobel Gradient + Threshold (τ) + Connected Components"

[8] ARROW → into:

[9] INSTANCE PSEUDO-LABEL IMAGE: The same Cityscapes scene with each individual
    object (car, person, bicycle) colored in a distinct random color.
    Background is black. Each car is a different color, each person a different
    color. Label: "Instance Pseudo Label"

=== SECTION 1b: SEMANTIC PSEUDO LABELING (bottom row, left to right) ===

[1] INPUT: Same RGB image as 1a (or a copy with dashed border indicating
    shared input). Label: "RGB Image"

[2] ARROW → into:

[3] PROCESSING BOX: White rounded rectangle.
    Text: "DINOv2 ViT-B/14 + CAUSE-TR (90-dim)" with lock icon (frozen).

[4] ARROW → into:

[5] PROCESSING BOX: White rounded rectangle.
    Text: "K-Means (k=80)"

[6] ARROW → into:

[7] SEMANTIC PREDICTION IMAGE: The same Cityscapes scene with each semantic
    class in a distinct color (road=purple, sidewalk=pink, building=gray,
    vegetation=green, sky=blue, car=dark blue, person=red).
    Label: "Semantic Predictions"

[8] DASHED DIAGONAL ARROW going UP from semantic predictions to the Sobel box
    in Section 1a, labeled "thing-class masks" — this shows that semantic
    labels tell the instance pipeline which pixels are "things" to split.

=== SECTION 1c: PANOPTIC ASSEMBLY (right column, dark gray background) ===

[1] ARROW from Instance Pseudo Label (1a) → into:

[2] PROCESSING BOX: White rounded rectangle on dark background.
    Text: "Stuff/Things Classifier" (uses depth-split ratio)

[3] ARROW down to:

[4] PROCESSING BOX: White rounded rectangle.
    Text: "Align (instance-first merge)"

[5] ARROWS from both Instance PL and Semantic PL converging into Align box.

[6] Three output images stacked vertically on the right edge (white text
    labels since dark background):
    - Top: "Instance Pseudo Label" — colored instance masks
    - Middle: "Panoptic Pseudo Label" — combined segmentation showing
      both stuff regions AND individual thing instances in distinct colors
    - Bottom: "Semantic Pseudo Label" — class-colored semantic map

=== STYLE GUIDELINES ===
- Light gray (#ECECEC) background for sections 1a and 1b
- Dark gray (#666666) background for section 1c
- White boxes with thin black borders for processing steps
- Lock/snowflake icons next to frozen model names
- Real photographic Cityscapes images embedded (urban driving scenes)
- Depth map in viridis/plasma colormap (warm=near, cool=far)
- Instance labels use random bright distinct colors per object on black bg
- Semantic labels use standard class colors (road=purple, sky=blue, etc.)
- Thin black arrows with arrowheads connecting all stages
- Yellow (#FFD54F) section header badges: "1a: Instance Pseudo Labeling",
  "1b: Semantic Pseudo Labeling", "1c: Panoptic Assembly"
- Minimal text — labels are short (2-4 words max per line)
- Flat design, no shadows or gradients on boxes
- Professional sans-serif typography

=== REFERENCE ASCII LAYOUT ===

┌─ 1a: Instance Pseudo Labeling ──────────────────────────────┐ ┌─ 1c: Assembly ──────┐
│                                                              │ │                      │
│ [RGB] ──→ [DAv3 🔒] ──→ [Depth Map] ──→ [Sobel+CC] ──→ [Inst PL] ──→ [Classifier]  │
│  img        frozen         colormap       threshold     colored    │      │          │
│                                                              │      ▼          │
│                                              ┌───── thing masks ──┐ [Align]       │
│                                              │      (dashed)      │    │          │
└──────────────────────────────────────────────┼───────────────────┘    ▼          │
                                               │              │  [Panoptic PL]   │
┌─ 1b: Semantic Pseudo Labeling ───────────────┼──────────────┐  [Semantic PL]   │
│                                              │              │  [Instance PL]   │
│ [RGB] ──→ [DINOv2+CAUSE 🔒] ──→ [K-Means] ──→ [Semantic PL] ──────────────────┘
│  img        frozen                k=80        class-colored  │
│                                                              │
└──────────────────────────────────────────────────────────────┘
```