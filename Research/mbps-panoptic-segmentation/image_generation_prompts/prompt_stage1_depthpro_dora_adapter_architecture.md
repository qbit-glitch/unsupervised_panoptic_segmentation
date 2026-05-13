Create a professional ML research paper architectural diagram showing the Apple DepthPro foundation model with injected DoRA adapters for self-supervised depth boundary refinement in a panoptic segmentation pipeline. Clean, minimal, publication-quality style similar to CVPR/NeurIPS paper figures. White background, no clutter. Flat design with distinct color coding per encoder and pipeline stage.

=== LAYOUT ===
Three horizontal sections stacked vertically, separated by thin dashed lines:
- Section A (top): "ADAPTER-INJECTED ARCHITECTURE" (blue accent #1565C0)
- Section B (middle): "SELF-SUPERVISED TRAINING" (green accent #2E7D32)
- Section C (bottom): "INFERENCE: DEPTH → INSTANCES" (orange accent #E65100)
Overall aspect ratio: wide (roughly 16:9)

=== SECTION A: ADAPTER-INJECTED ARCHITECTURE (top) ===
Left to right flow showing the modified DepthPro backbone:

[1] INPUT BOX (far left):
    - Small real Cityscapes street scene thumbnail (urban driving scene)
    - Label: "RGB Image (3×H×W)"
    - Sub-label: "Cityscapes unlabeled train"

[2] LONG ARROW → pointing right into:

[3] DUAL-ENCODER BACKBONE (center, large):
    - Two tall vertical rectangular slabs side by side, each representing
      24 transformer blocks stacked vertically
    - Left slab: PATCH ENCODER (blue tint #1565C0, label above)
    - Right slab: IMAGE ENCODER (purple tint #7B1FA2, label above)
    - Each slab shows internal division: upper portion (lighter shade)
      labeled "Blocks 0–17 (Early)", lower portion (darker shade)
      labeled "Blocks 18–23 (Late)"

    Inside each slab, show tiny adapter badges:
    - Early portion: small green badges "DoRA r=4" next to "Q" and "V"
      labels; gray "FROZEN" next to "K", "Proj", "FC1", "FC2"
    - Late portion: green badges "DoRA r=4" next to ALL six: "Q", "K",
      "V", "Proj", "FC1", "FC2"

    Between the two slabs: small double-headed arrow labeled
    "Cross-Attention / Fusion"

[4] ARROW down from dual-encoder into:

[5] DECODER HEAD BOX (below center, small):
    - Gray rounded rectangle with lock/snowflake icon
    - Label: "DPT-Style Decoder [FROZEN]"

[6] ARROW down into:

[7] FOV ENCODER BOX (below decoder, small):
    - Gray rounded rectangle with lock/snowflake icon
    - Label: "FOV Encoder (DINOv2-Large, 24L)"
    - Sub-label: "[NO ADAPTERS — COMPLETELY FROZEN]"
    - Color: gray (#9E9E9E)

[8] ARROW down to:

[9] OUTPUT DEPTH MAP (bottom-left of section):
    - A colorful depth map visualization (viridis/plasma colormap)
      of the same Cityscapes scene — nearby objects bright/warm,
      far objects dark/cool
    - Label: "Adapted Metric Depth (1×H×W)"

[10] PARAMETER BADGE (floating right of the dual-encoder):
    - Prominent rounded rectangle with green background (#2E7D32)
      and white bold text:
      "~1.66M TRAINABLE"
      "~1B FROZEN"
      "0.17% of model"
    - Small connector line pointing to the dual-encoder slabs

=== SECTION B: SELF-SUPERVISED TRAINING (middle) ===
Circular/loop layout showing student-teacher setup:

[1] TEACHER MODEL (top-right of this section):
    - Same dual-slab visualization as Section A but in orange tint (#E65100)
    - Label: "TEACHER: DepthPro (ORIGINAL)"
    - Lock/snowflake icon indicating frozen
    - Sub-label: "ALL ~1B params frozen"

[2] SHARED INPUT (center-top, between student and teacher):
    - Small Cityscapes thumbnail
    - Label: "Shared Input: RGB Image"
    - Two arrows branching: one to Teacher, one to Student

[3] STUDENT MODEL (bottom-left of this section):
    - Same dual-slab visualization as Section A in green tint (#2E7D32)
    - Label: "STUDENT: DepthPro + DoRA Adapters"
    - Sub-label: "~1.66M params trainable"

[4] THREE LOSS BOXES (center, stacked vertically between student and teacher):
    - Top box (light blue): "L_dist = MSE(student, teacher.detach())"
      with small label "w₁ = 1.0"
    - Middle box (light purple): "L_rank = MarginRankingLoss(pairs, margin=0.1)"
      with small label "w₂ = 0.1"
    - Bottom box (light orange): "L_si = Scale-Invariant Log Loss"
      with small label "w₃ = 0.5"
    - Each box has arrows from both Student Depth and Teacher Depth
      converging into it

[5] COMBINED LOSS BOX (below the three losses):
    - Slightly larger rounded rectangle
    - Label: "L_total = w₁·L_dist + w₂·L_rank + w₃·L_si"
    - Arrow down to:

[6] OPTIMIZER BOX (bottom-center):
    - Label: "AdamW (lr=1e-4, WD=1e-4)"
    - Sub-label: "Only adapter params enter optimizer"
    - Curved arrow going LEFT from optimizer back to Student

=== SECTION C: INFERENCE — DEPTH TO INSTANCES (bottom) ===
Left to right pipeline:

[1] INPUT (far left):
    - Same Cityscapes RGB thumbnail as Section A
    - Label: "RGB Image"

[2] ARROW → into:

[3] ADAPTED DEPTHPRO BOX:
    - Green-tinted rounded rectangle
    - Label: "DepthPro + DoRA (trained)"
    - Sub-label: "freeze_non_adapter_params()"

[4] ARROW → into:

[5] DEPTH MAP OUTPUT:
    - Viridis/plasma colormap depth map thumbnail
    - Label: "Adapted Depth Map"

[6] ARROW → into:

[7] SOBEL + THRESHOLD BOX:
    - White rounded rectangle with thin border
    - Label: "Sobel Edge Detection"
    - Sub-label: "grad_mag = √(gx² + gy²)"
    - Prominent badge below: "τ = 0.20" (red background #C62828, white text)

[8] ARROW → into:

[9] CONNECTED COMPONENTS BOX:
    - White rounded rectangle
    - Label: "Connected Components"
    - Sub-label: "per semantic 'thing' class"
    - Small text list: "person, car, rider, truck, bus, train, motorcycle, bicycle"

[10] ARROW → into:

[11] POST-PROCESS BOX:
    - White rounded rectangle
    - Label: "Dilate + Reclaim (3 iters) + Area Filter (min 1000 px)"

[12] ARROW → into:

[13] FINAL OUTPUT (far right):
    - Instance pseudo-label visualization: the same Cityscapes scene with
      each individual object (car, person, bicycle) colored in a distinct
      random bright color on black background
    - Label: "Instance Pseudo-Labels"
    - Sub-label: "masks[N×H×W], classes[N], scores[N]"

=== STYLE GUIDELINES ===
- Clean white background for the overall figure
- Section backgrounds: very light tint corresponding to accent color
  (e.g., light blue #E3F2FD for Section A, light green #E8F5E9 for B,
   light orange #FFF3E0 for C)
- Thin black arrows with arrowheads connecting all stages
- Neural network encoders shown as tall vertical stacks of small
  rectangular blocks (24 blocks), perspective 3D slab style
- Green "DoRA r=4" badges are small rounded pills overlaying the blocks
- Gray "FROZEN" badges are small muted pills
- Color coding:
  • Patch Encoder elements: blue (#1565C0)
  • Image Encoder elements: purple (#7B1FA2)
  • FOV Encoder / frozen parts: gray (#9E9E9E) with lock icon
  • Student network: green (#2E7D32)
  • Teacher network: orange (#E65100)
  • Loss boxes: distinct pastel tints (blue #BBDEFB, purple #E1BEE7, orange #FFE0B2)
  • Parameter badge: green (#2E7D32) with white text
  • Threshold τ badge: red (#C62828) with white text
- Real image thumbnails embedded at input and output stages:
  Cityscapes urban driving scenes with depth viridis maps and
  colorful instance mask overlays
- Depth maps: viridis/plasma colormap (warm=near, cool=far)
- Instance labels: random bright distinct colors per object on black background
- Rounded rectangle boxes with thin borders (1px)
- Professional sans-serif typography (consistent sizing:
  section headers 14pt, labels 11pt, sub-labels 9pt)
- No gradients or drop shadows — flat design
- Minimal text; use icons/visuals over words where possible
- Similar style to CVPR 2025/NeurIPS 2025 paper figures

=== REFERENCE ASCII LAYOUT ===

┌─ A: ADAPTER-INJECTED ARCHITECTURE ───────────────────────────────────────────┐
│                                                                              │
│  [RGB] ──→  ┌─ PATCH ENCODER ─┐ ┌─ IMAGE ENCODER ─┐                        │
│   img        │  (Blue #1565C0) │ │ (Purple #7B1FA2)│                        │
│              │  Blocks 0-17:   │ │  Blocks 0-17:   │                        │
│              │   Q→[DoRA r=4]  │ │   Q→[DoRA r=4]  │                        │
│              │   K→[FROZEN]    │ │   K→[FROZEN]    │                        │
│              │   V→[DoRA r=4]  │ │   V→[DoRA r=4]  │                        │
│              │   Proj→[FROZEN] │ │   Proj→[FROZEN] │                        │
│              │   FC1→[FROZEN]  │ │   FC1→[FROZEN]  │                        │
│              │   FC2→[FROZEN]  │ │   FC2→[FROZEN]  │                        │
│              │  Blocks 18-23:  │ │  Blocks 18-23:  │                        │
│              │   Q→[DoRA r=4]  │ │   Q→[DoRA r=4]  │     ┌──────────────┐   │
│              │   K→[DoRA r=4]  │ │   K→[DoRA r=4]  │     │ ~1.66M       │   │
│              │   V→[DoRA r=4]  │ │   V→[DoRA r=4]  │     │ TRAINABLE    │   │
│              │   Proj→[DoRA]   │ │   Proj→[DoRA]   │     │ ~1B FROZEN   │   │
│              │   FC1→[DoRA]    │ │   FC1→[DoRA]    │     │ 0.17%        │   │
│              │   FC2→[DoRA]    │ │   FC2→[DoRA]    │     └──────────────┘   │
│              └─────────────────┘ └─────────────────┘                        │
│                       ↕ Fusion / Cross-Attn                                 │
│                              │                                              │
│                              ▼                                              │
│              ┌─────────────────────────────┐                                │
│              │  DPT-Style Decoder [FROZEN] │  ← lock icon                   │
│              └─────────────────────────────┘                                │
│                              │                                              │
│                              ▼                                              │
│              ┌─────────────────────────────┐                                │
│              │  FOV Encoder (24L) [FROZEN] │  ← gray, NO adapters           │
│              └─────────────────────────────┘                                │
│                              │                                              │
│                              ▼                                              │
│                       [Adapted Depth Map]                                   │
└──────────────────────────────┬───────────────────────────────────────────────┘
                               │ checkpoint / shared input
┌─ B: SELF-SUPERVISED TRAINING ┼──────────────────────────────────────────────┐
│                              │                                              │
│         ┌──────── TEACHER ─────────┐                                        │
│         │ DepthPro (ORIGINAL)      │  ← orange, ALL frozen                  │
│         │ ~1B params, lock icon    │                                        │
│         └───────────┬──────────────┘                                        │
│                     │ Teacher Depth                                        │
│    Shared Input     │                                                      │
│    [RGB img] ───────┼────────────────→ ┌─────────────────┐                  │
│         │           │                  │ L_dist = MSE    │ w₁=1.0           │
│         │           │ Student Depth    │ L_rank = Margin │ w₂=0.1           │
│         └────────→ STUDENT ──────┬──→ │ L_si = ScaleInv │ w₃=0.5           │
│           DepthPro + DoRA        │     └─────────────────┘                  │
│           ~1.66M trainable       │              │                           │
│           green                  │              ▼                           │
│                                  │     ┌─────────────────┐                  │
│                                  │     │ L_total = Σw·L  │                  │
│                                  │     └─────────────────┘                  │
│                                  │              │                           │
│                                  │              ▼                           │
│                                  │     AdamW (lr=1e-4)                      │
│                                  │     only adapter params                  │
│                                  └──────────────────────────────────────    │
└─────────────────────────────────────────────────────────────────────────────┘
┌─ C: INFERENCE — DEPTH → INSTANCES ──────────────────────────────────────────┐
│                                                                              │
│  [RGB] ──→ [DepthPro+DoRA] ──→ [Depth Map] ──→ [Sobel+τ=0.20] ──→ [CC] ──→ │
│   img        trained, frozen    viridis          grad_mag > 0.20   per class │
│                                                    │              thing      │
│                                                    │              mask       │
│                                                    ▼                         │
│                                              [Dilate+Reclaim(3)] ──→ [Inst] │
│                                               + Area Filter(1000)     PL     │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
