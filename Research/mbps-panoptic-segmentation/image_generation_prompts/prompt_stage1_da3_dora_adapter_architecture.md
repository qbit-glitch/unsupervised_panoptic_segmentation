# Image Generation Prompt: DA3 DoRA Adapter Architecture for MBPS Stage 1

```md
Create a professional ML research paper architectural diagram showing the Depth Anything V3 (DA3) DoRA adapter architecture for unsupervised depth-guided instance pseudo-label generation. Clean, minimal, publication-quality style similar to CVPR/NeurIPS paper figures. White background, no clutter, flat design with crisp edges.

=== LAYOUT ===
Wide landscape figure (roughly 16:9) with three horizontal sections stacked vertically,
plus a right-side comparison panel spanning the top two sections.

- Section 1 (top, ~28% height): "DA3 Encoder with DoRA Adapters" (teal accent #00897B)
- Section 2 (middle, ~32% height): "Self-Supervised Adapter Training" (blue accent #1565C0 for student)
- Section 3 (bottom, ~40% height): "Depth-Guided Instance Pseudo-Label Pipeline" (green accent #2E7D32)
- Right Panel (spanning sections 1-2, ~18% width): "Comparison" with mini-table

=== SECTION 1: DA3 ENCODER WITH DoRA ADAPTERS (top) ===
Left-to-right flow:

[1] INPUT BOX (left edge):
    - A real Cityscapes urban street scene thumbnail (photograph, thin gray border)
    - Label: "RGB Image  (B, 3, H, W)"
    - Small tag: "Cityscapes leftImg8bit"

[2] YELLOW WARNING BADGE (above the arrow, floating):
    - Rounded rectangle with yellow fill (#FFD54F) and bold black text
    - Text: "Custom API: depth_anything_3.api.DepthAnything3"
    - Subtext in smaller font with strikethrough: "NOT transformers.AutoModel"
    - Small icon: generic code brackets or a plug icon indicating non-standard interface

[3] LONG ARROW → pointing right into:

[4] DA3 INTERNAL PREPROCESSING box:
    - White rounded rectangle with thin teal border
    - Text: "Preprocess (resize, normalize, patchify)"
    - Small lock/snowflake icon with label "FROZEN"

[5] ARROW → into:

[6] VIT ENCODER visualization (center, largest element in this section):
    - Show as a wide horizontal stack of rectangular blocks representing transformer blocks
    - Background shading: light teal (#E0F2F1)
    - Label at top: "ViT Encoder  ~307M params  DINOv2/DINOv3-Large  dim=1024"
    - The encoder is divided into two sub-regions by a subtle dashed vertical line:

    [6a] EARLY BLOCKS region (left portion, blocks 0-5):
        - 6 small rectangular blocks in a row, teal fill (#00897B)
        - Each block labeled "Block i" at top
        - Inside each block: a small sub-box labeled "ATTN" with "qkv" highlighted
          by a yellow (#FFD54F) DoRA badge arrow pointing to it
        - Label below region: "Early (0-5): qkv only"
        - Small text: "Preserve low-level features"

    [6b] LATE BLOCKS region (right portion, blocks 6-N):
        - Several blocks continuing the row, darker teal (#00695C)
        - Each block taller than early blocks to show more layers
        - Inside each block: "ATTN" with "qkv" and "proj" both highlighted by yellow
          DoRA badges; below that "MLP" with "fc1" and "fc2" highlighted by yellow
          DoRA badges
        - Label below region: "Late (6-N): qkv + proj + mlp.fc1 + mlp.fc2"
        - Small text: "Full adaptation"

    [6c] GENERIC INJECTION BADGE (floating above the encoder, spanning both regions):
        - Yellow rounded pill (#FFD54F) with black text
        - Text: "Generic Injection: named_modules() walker"
        - Subtext: "Fallback for non-HF architecture"
        - Small dashed arrow pointing down at the encoder blocks

[7] ARROW → into:

[8] DPT DECODER HEAD box:
    - White rounded rectangle with thin gray border
    - Text: "DPT Decoder (feature fusion + upsample)"
    - Lock/snowflake icon with label "FROZEN — no adapters"

[9] ARROW → to:

[10] OUTPUT (right edge):
    - A colorful depth map visualization (viridis/plasma colormap) of the same
      Cityscapes scene — nearby objects bright/warm, far objects dark/cool
    - Label: "Relative Depth Map  (B, H, W)  ∈ [0,1]"

=== SECTION 2: SELF-SUPERVISED ADAPTER TRAINING (middle) ===
Parallel/circular student-teacher flow:

[1] INPUT (top-center of this section):
    - White rounded rectangle
    - Label: "Unlabeled RGB Images"
    - Sub-label: "Cityscapes train split"
    - A small Cityscapes thumbnail

[2] SPLIT ARROW going down and branching left + right:

[3] STUDENT MODEL (left side, blue-tinted):
    - Vertical stack of layer slabs tinted blue (#1565C0)
    - Label at top: "STUDENT" in bold blue
    - Sub-label: "DA3 + DoRA adapters"
    - Small badges on encoder layers: "lora_A", "lora_B", "magnitude" in yellow pills
    - Decoder portion grayed out with "FROZEN" label
    - Output arrow down labeled "D_student"

[4] TEACHER MODEL (right side, orange-tinted):
    - Same layer slab shape but tinted orange (#E65100)
    - Label at top: "TEACHER" in bold orange
    - Sub-label: "DA3 (frozen, no adapters)"
    - Lock/snowflake icon on every layer
    - Output arrow down labeled "D_teacher (detach)"

[5] LOSS COMPUTATION BOX (center-bottom, spanning between student and teacher outputs):
    - Large white rounded rectangle with thin black border
    - Title: "Loss Computation"
    - Three horizontal loss bars stacked inside:

      [5a] Blue bar (#1565C0): "L_distill = MSE(D_s, D_t.detach())"  |  w = 1.0
           Small text below: "Keep student close to pretrained"

      [5b] Green bar (#2E7D32): "L_rank = MarginRanking(pixel pairs)"  |  w = 0.1
           Small text below: "Preserve ordinal depth relationships"

      [5c] Purple bar (#7B1FA2): "L_si = ScaleInvariant(D_s, D_t)"  |  w = 0.5
           Small text below: "Handle unknown absolute scale"

    - Below the three bars: "L_total = Σ w_i · L_i"
    - Arrow pointing down labeled "Gradients → STUDENT adapters only"

[6] OPTIMIZER BOX (below loss box):
    - Small rounded rectangle
    - Text: "AdamW  |  LR = 1e-4  |  CosineAnnealing"
    - Subtext in smaller font: "~1.2M–1.5M trainable params (r=4)  |  ~0.4–0.5% of total"

[7] TRAINABLE PARAMS BADGE (floating near student, bottom-left):
    - Teal pill badge
    - Text: "Trainable: lora_A (4,1024), lora_B (1024,4), magnitude (1024,1)"
    - Subtext: "9,216 params per layer × ~130-160 layers"

=== SECTION 3: DEPTH-GUIDED INSTANCE PSEUDO-LABEL PIPELINE (bottom) ===
Left-to-right flow:

[1] INPUT (left edge):
    - Two stacked thumbnails:
      - Top: the colorful adapted depth map (same as Section 1 output)
      - Bottom: a semantic pseudo-label map (class-colored, from DINOv3)
    - Labels: "Adapted Depth D" and "Semantic S (DINOv3)"

[2] ARROW → into:

[3] DEPTH PROCESSING column (three stacked boxes):
    [3a] "Gaussian Blur  σ = 1.0"  →  [3b] "Sobel Gradient  |∇D|"  →  [3c] "Threshold"

[4] THRESHOLD BOX (highlighted, most prominent in this column):
    - White rounded rectangle with thick green border (#2E7D32)
    - Bold text: "τ = 0.03"
    - Subtext: "Empirically optimal for DA3"
    - Small callout bubble: "DA3 sharper edges → low τ works"

[5] ARROW → into:

[6] EDGE MASK visualization:
    - Binary edge mask image (white edges on black background) showing
      crisp object boundaries around cars and pedestrians
    - Label: "Edge Mask E = |∇D| > τ"

[7] ARROW → into:

[8] INSTANCE GENERATION BOX:
    - White rounded rectangle
    - Title: "Per Thing-Class Connected Components"
    - Bulleted steps inside:
      • "M_c = (S == c) for c ∈ {person, car, truck, bus, ...}"
      • "Remove edges: M'_c = M_c & ~E"
      • "Connected Components → CC-1, CC-2, ..."
      • "Dilation reclamation (3 iters)"
      • "Filter: area ≥ 100 px"
    - Small thing-class icon strip: person, car, truck, bus, bicycle silhouettes

[9] ARROW → into:

[10] OUTPUT (right edge):
    - A real Cityscapes scene with each individual object instance colored in a
      distinct bright random color (car=red, person=yellow, bicycle=cyan, etc.)
      on a black background
    - Label: "Instance Pseudo-Labels"
    - Sub-label: "NPZ: masks, scores, classes"

=== RIGHT COMPARISON PANEL (spanning sections 1-2) ===
[1] Mini-table with 4 rows + header, clean grid lines:

    ┌─────────────┬──────────────┬─────┬───────────┬────────┐
    │ Model       │ API Type     │ τ   │ PQ_things │ PQ     │
    ├─────────────┼──────────────┼─────┼───────────┼────────┤
    │ SPIdepth    │ Standard     │0.15 │ 17.30     │ —      │
    │ DA2-Large   │ HF AutoModel │0.03 │ 20.20     │ ~26.5  │
    │ DA3 (Ours)  │ Custom API   │0.03 │ 20.90     │ 27.37  │
    └─────────────┴──────────────┴─────┴───────────┴────────┘

    - The DA3 row has a bold green left border (#2E7D32) and "BEST" badge in green
    - Below table: "+3.5% PQ_things vs DA2-Large  |  +0.7 at same τ"

[2] PQ HIGHLIGHT BADGE (below the table):
    - Large rounded rectangle with teal background (#00897B) and white bold text
    - Text: "PQ = 27.37"
    - Subtext: "PQ_things = 20.90"
    - Small label: "Cityscapes val"

=== STYLE GUIDELINES ===
- Clean white background for the entire figure
- Flat design — no drop shadows, no gradients on boxes (solid fills only)
- Neural network layers shown as colorful 3D rectangular slabs
  (perspective view, like looking at a bookshelf from an angle)
- Thin black arrows with triangular arrowheads connecting all stages
- Rounded rectangle boxes with thin borders (1-2px)
- Color scheme:
  • Teal (#00897B) for DA3 encoder elements, architecture blocks
  • Dark teal (#00695C) for late blocks in tiered visualization
  • Yellow (#FFD54F) for generic injection badges, DoRA layer highlights, warning pills
  • Blue (#1565C0) for Student model and distillation loss
  • Orange (#E65100) for Teacher model
  • Green (#2E7D32) for ranking loss, threshold highlights, best-result badges
  • Purple (#7B1FA2) for scale-invariant loss
  • Light gray (#ECECEC) for frozen/background elements
  • Black (#212121) for all text and borders
- Real image thumbnails embedded at input and output stages:
  • Cityscapes RGB photographs at inputs
  • Depth maps in viridis/plasma colormap
  • Instance pseudo-labels with random bright distinct colors per object on black bg
- Professional sans-serif typography, consistent sizing hierarchy
- Lock/snowflake icons next to all frozen components
- Section headers: bold, all-caps, with colored underline matching section accent
- Minimal text — short labels (2-5 words), use icons and visual badges over words
- Similar style to CVPR 2025/NeurIPS 2025 paper figures

=== REFERENCE ASCII LAYOUT ===

┌─ SECTION 1: DA3 ENCODER WITH DoRA ADAPTERS ─────────────────────────────┐ ┌─ COMPARISON ─────────┐
│                                                                         │ │                      │
│  [RGB] ──→ [Preprocess 🔒] ──→ [══════════════════════════════════] ──→ [Depth Map] │ │  Model    │τ│PQth│ PQ │
│   img       FROZEN               VIT ENCODER  ~307M  dim=1024           viridis    │ │  SPIdepth │0.15│17.30│ —  │
│                                  │  Early │  Late │                    │ │  DA2-L    │0.03│20.20│26.5│
│    ┌─ "NOT HF AutoModel" ──┐    │ 0-5    │ 6-N  │                    │ │  DA3(Ours)│0.03│20.90│27.37│
│    │ Custom API:            │    │ qkv◄───┤ qkv◄─┤  ◄── DoRA badges   │ │  ─────────┴─┴────┴────┘
│    │ depth_anything_3.api   │    │ frozen │ proj◄┤                    │ │  [ PQ = 27.37      ] │
│    │ .DepthAnything3        │    │        │ fc1◄─┤  ◄── DoRA badges   │ │  [ PQ_th=20.90 BEST] │
│    └────────────────────────┘    │        │ fc2◄─┤                    │ │                      │
│         ▲ "Generic Injection:    │        │      │                    │ │                      │
│           named_modules() walker"│        │      │                    │ │                      │
│                                  └────────┴──────┘                    │ └──────────────────────┘
│                                         │                             │
│                                  [DPT Decoder 🔒]                     │
│                                   FROZEN — no adapters                │
└─────────────────────────────────────────────────────────────────────────┘
┌─ SECTION 2: SELF-SUPERVISED ADAPTER TRAINING ───────────────────────────┘
│                                                                         │
│                    [ Unlabeled RGB Images ]                             │
│                     Cityscapes train split                              │
│                             │                                           │
│            ┌────────────────┼────────────────┐                          │
│            │                │                │                          │
│            ▼                │                ▼                          │
│    ┌──────────────┐         │        ┌──────────────┐                   │
│    │   STUDENT    │         │        │   TEACHER    │                   │
│    │  DA3 + DoRA  │         │        │  DA3 (frozen)│                   │
│    │  [████] blue │         │        │  [████]orange│                   │
│    │  lora_A/B/mag│         │        │  🔒🔒🔒🔒🔒   │                   │
│    │  D_student   │         │        │  D_teacher   │                   │
│    └──────┬───────┘         │        └──────┬───────┘                   │
│           │                 │               │                           │
│           └─────────────────┼───────────────┘                           │
│                             ▼                                           │
│           ┌─────────────────────────────────┐                           │
│           │      LOSS COMPUTATION           │                           │
│           │  ┌───────────────────────────┐  │                           │
│           │  │ L_distill (blue)  w=1.0   │  │                           │
│           │  │ MSE(D_s, D_t.detach())    │  │                           │
│           │  ├───────────────────────────┤  │                           │
│           │  │ L_rank    (green) w=0.1   │  │                           │
│           │  │ MarginRanking(pairs)      │  │                           │
│           │  ├───────────────────────────┤  │                           │
│           │  │ L_si      (purple) w=0.5  │  │                           │
│           │  │ ScaleInvariant(D_s, D_t)  │  │                           │
│           │  └───────────────────────────┘  │                           │
│           │  L_total = Σ w_i · L_i          │                           │
│           │  Grads → STUDENT adapters only  │                           │
│           └─────────────────────────────────┘                           │
│                             │                                           │
│                    [AdamW  LR=1e-4  Cosine]                             │
│                    ~1.2M–1.5M params (r=4)                              │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
┌─ SECTION 3: DEPTH-GUIDED INSTANCE PSEUDO-LABEL PIPELINE ────────────────┘
│                                                                         │
│  [Depth D]  [Semantic S]                                                │
│   adapted    DINOv3                                                     │
│      │         │                                                        │
│      ▼         │                                                        │
│  [Gaussian Blur σ=1.0]                                                  │
│      │                                                                    │
│      ▼                                                                    │
│  [Sobel Gradient |∇D|]                                                   │
│      │                                                                    │
│      ▼                                                                    │
│  ┌─────────────────────┐                                                │
│  │   τ = 0.03          │  ◄── OPTIMAL (thick green border)              │
│  │   DA3 sharper edges │                                                │
│  └─────────────────────┘                                                │
│      │                                                                    │
│      ▼                                                                    │
│  [Edge Mask E = |∇D|>τ]  ──→  [Per Thing-Class CC]  ──→  [Instance PL] │
│   white edges on black          • M_c = (S==c)            colored masks │
│                                 • M'_c = M_c & ~E         NPZ output    │
│                                 • Connected Components                  │
│                                 • Dilation (3 iters)                    │
│                                 • Filter area ≥ 100 px                  │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```
