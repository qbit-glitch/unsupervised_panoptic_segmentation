```md
Create a professional ML research paper architectural diagram showing the Depth Anything V2 Large (DA2-Large) encoder with tiered DoRA adapters, student-teacher self-supervised training, and depth-to-instance-pseudo-label inference pipeline. Clean, minimal, publication-quality style similar to CVPR/NeurIPS paper figures. White background, no clutter, flat design.

=== LAYOUT ===
Three sections arranged in a wide landscape (roughly 16:9):
- Section A (top row, ~45% height): "ADAPTER-INJECTED ARCHITECTURE" — vertical encoder stack with horizontal flow into decoder
- Section B (bottom-left, ~55% width of bottom row): "STUDENT-TEACHER TRAINING" — parallel model boxes with loss computation
- Section C (bottom-right, ~45% width of bottom row): "INFERENCE PIPELINE" — linear processing chain
A thin dashed separator line between top and bottom rows.

=== SECTION A: ADAPTER-INJECTED ARCHITECTURE (top row) ===

Left to right flow:

[1] INPUT BOX (far left):
    - Small real Cityscapes street scene thumbnail (urban driving scene)
    - Label: "RGB Image (H × W × 3)"
    - Below: small badge "HF AutoImageProcessor"

[2] LONG ARROW → pointing right into:

[3] VERTICAL ENCODER STACK (center-left, tall rectangular region):
    - Title bar at top: "DINOv2-Large Encoder (backbone.encoder.layer)"
    - Subtitle: "24 blocks × 1024-dim × 16 heads"
    - Show 24 horizontal block layers stacked vertically, each as a thin colored rectangle
    
    Blocks 0–17 (early tier, light blue fill #81D4FA, ~3/4 of stack height):
    - Each block shown as a horizontal rounded rectangle
    - Left side of each block: small blue badges "Q+[DoRA]", "V+[DoRA]"
    - Right side of each block: faint gray text "K · proj · fc1 · fc2" (frozen)
    - Label on left margin: "Early: 0–17" with light blue (#81D4FA) accent bar
    
    Blocks 18–23 (late tier, dark blue fill #1565C0, ~1/4 of stack height):
    - Each block shown as a horizontal rounded rectangle, slightly thicker/darker
    - Badges on each block: "Q+[DoRA]", "K+[DoRA]", "V+[DoRA]", "proj+[DoRA]", "fc1+[DoRA]", "fc2+[DoRA]"
    - Label on left margin: "Late: 18–23" with dark blue (#1565C0) accent bar
    
    - Inside each block, show a mini internal structure:
      Left half: "attention.attention.query / .key / .value / .output.dense"
      Right half: "mlp.fc1 (1024→4096) / mlp.fc2 (4096→1024)"
    
    - A floating callout box near the encoder stack:
      "Trainable: ~829K params"
      "Frozen: ~299M params"
      "Ratio: 0.28%"
      With a small pie-chart icon showing the tiny trainable slice

[4] ARROW → pointing right (labeled "multi-scale features [1/4, 1/8, 1/16, 1/32]") into:

[5] DPT DECODER BOX (center-right):
    - Gray fill (#BDBDBD) with snowflake/lock icon indicating frozen
    - Label: "DPT Decoder Head — FROZEN"
    - Internal sub-boxes: "Reassemble (1/4)", "Reassemble (1/8)", "Reassemble (1/16)", "Fusion + Refine"
    - Arrow down to "1×1 Conv"
    - Small badge: "adapt_decoder = False"

[6] ARROW → pointing right into:

[7] OUTPUT DEPTH MAP (far right):
    - A real depth map visualization (viridis/plasma colormap) of a Cityscapes scene
    - Label: "Relative Depth Map (H/14 × W/14)"
    - Small text below: "❌ No absolute metric scale"

[8] HF INTEGRATION BADGE (floating below the encoder stack):
    - Rounded rectangle with HuggingFace logo (🤗)
    - Text: "AutoModelForDepthEstimation.from_pretrained()"
    - "HF Transformers Compatible"

=== SECTION B: STUDENT-TEACHER TRAINING (bottom-left) ===

Two parallel boxes with converging arrows to a central loss box:

[1] STUDENT MODEL BOX (left, green accent #43A047):
    - Same colorful layer-slab visualization as encoder but with green tint
    - Label: "STUDENT"
    - Sub-labels:
      "Encoder: DoRA adapters ACTIVE"
      "Decoder: FROZEN"
      "Trainable: ~829K params"
    - Small green arrow going down from this box labeled "gradients flow here only"

[2] TEACHER MODEL BOX (right, orange accent #FB8C00):
    - Same layer-slab visualization with orange tint
    - Label: "TEACHER (frozen copy)"
    - Sub-labels:
      "Encoder: NO adapters"
      "Decoder: FROZEN"
      "Trainable: 0 params"
    - Lock/snowflake icon

[3] INPUT CONVERGENCE (between student and teacher, at top):
    - Two thin arrows from a shared "RGB Image" thumbnail (small Cityscapes scene)
    - Left arrow: "+ augmentation" → Student
    - Right arrow: "no aug" → Teacher

[4] LOSS COMPUTATION BOX (center, below both models):
    - Large rounded rectangle with white fill and thin black border
    - Title: "Loss Computation"
    - Three labeled equations stacked vertically:
    
      ① DISTILLATION (MSE) — weight 1.0
        L_mse = ||D_student − D_teacher||²
      
      ② RELATIVE DEPTH RANKING — weight 0.1
        L_rank = MarginRankingLoss(sign(D_i − D_j), margin=0.1)
      
      ③ SCALE-INVARIANT (Eigen et al.) — weight 0.5
        L_si = (log D − log D*)² − λ(Σ(log D − log D*))²/n²
    
    - Bottom of box, bold:
      TOTAL = 1.0·L_mse + 0.1·L_rank + 0.5·L_si
    
    - Small config badges below the box:
      "AdamW, lr=1e-4", "Cosine Annealing", "Grad Clip max=1.0"

[5] CURVED GREEN ARROW from loss box back up to STUDENT box (left side):
    - Indicates gradient update loop
    - Only student adapters receive gradients

=== SECTION C: INFERENCE PIPELINE (bottom-right) ===

Left to right linear chain:

[1] INPUT: Small depth map thumbnail (same viridis colormap as Section A output)
    - Label: "Depth Map (.npy)"

[2] ARROW → into:

[3] SOBEL GRADIENT BOX:
    - White rounded rectangle
    - Text:
      "Sobel_x = cv2.Sobel(depth, dx=1, dy=0)"
      "Sobel_y = cv2.Sobel(depth, dx=0, dy=1)"
      "grad_mag = √(Sobel_x² + Sobel_y²)"

[4] ARROW → into:

[5] THRESHOLD BOX:
    - White rounded rectangle
    - Text: "Binary Mask = grad_mag > τ"
    - Large bold badge: "τ = 0.03" (Cityscapes optimal)
    - Small note: "(COCO: τ = 0.08)"

[6] ARROW → into:

[7] CONNECTED COMPONENTS BOX:
    - White rounded rectangle
    - Text:
      "CC on (1 − binary_mask)"
      "Each region = candidate instance"
      "Filter: min area 50–200 px"

[8] ARROW → into:

[9] OUTPUT INSTANCE PSEUDO-LABEL (far right):
    - Real instance segmentation visualization of a Cityscapes scene
    - Each object (car, person, bicycle) in a distinct random bright color on black background
    - Label: "Instance Pseudo-Label (.png)"
    - Bold metric badge: "PQ_things = 20.20%"
    - Small text: "(DA2-Large, Cityscapes)"

=== STYLE GUIDELINES ===
- Clean white background for the entire figure
- Color coding (STRICT):
  • HF-standard encoder layers: base blue (#2196F3)
  • Early tier DoRA blocks (0–17): light blue (#81D4FA)
  • Late tier DoRA blocks (18–23): dark blue (#1565C0)
  • Frozen DPT decoder: gray (#BDBDBD) with snowflake icon
  • Student model elements: green (#43A047)
  • Teacher model elements: orange (#FB8C00)
  • Loss computation box: white fill, thin black border
  • Inference processing boxes: white fill, thin black border
- Thin black arrows with arrowheads connecting all stages
- Neural network encoder blocks shown as horizontal rectangular slabs stacked vertically (like a sliced layer cake viewed from the side)
- DoRA badges as small rounded rectangles attached to the right edge of each block layer
- Real image thumbnails embedded at input and output stages (Cityscapes urban driving scenes)
- Depth map visualizations in viridis/plasma colormap (warm=near, cool=far)
- Instance pseudo-label uses random bright distinct colors per object on black background
- Professional sans-serif typography, consistent sizing hierarchy
- No gradients or drop shadows — flat design
- Section headers use bold text with colored underline bars
- Publication-quality, similar to CVPR 2025 / NeurIPS 2025 paper figures
- Minimal text in encoder blocks — use abbreviations and badges
- Lock/snowflake icons next to all frozen components

=== REFERENCE ASCII LAYOUT ===

┌─ SECTION A: ADAPTER-INJECTED ARCHITECTURE ─────────────────────────────────────────────────────────┐
│                                                                                                      │
│  [RGB] ──→ [████████████████████████████████████████████████████████████████] ──→ [Depth Map]      │
│  Cityscapes     DINOv2-Large Encoder (24 blocks)                                viridis            │
│                 ┌──────────────────────────────────────────────────────────┐                         │
│                 │  Early 0–17  [light blue]  Q+[DoRA] V+[DoRA] │ K·proj·fc1·fc2 (frozen)            │
│                 │  attention.attention.query/.key/.value/.output.dense                        │        │
│                 │  mlp.fc1 (1024→4096)  mlp.fc2 (4096→1024)                               │        │
│                 │  Late 18–23  [dark blue]  Q+[DoRA] K+[DoRA] V+[DoRA] proj+[DoRA] fc1+[DoRA] fc2+[DoRA] │
│                 └──────────────────────────────────────────────────────────┘                         │
│                                   │                                                                  │
│                                   ▼  multi-scale [1/4, 1/8, 1/16, 1/32]                              │
│                 ┌─────────────────────────────────────┐                                              │
│                 │  DPT Decoder Head ❄️ FROZEN         │                                              │
│                 │  Reassemble → Fusion + Refine → 1×1 │                                              │
│                 └─────────────────────────────────────┘                                              │
│                                                                                                      │
│  ┌────────────────────────┐  🤗 AutoModelForDepthEstimation.from_pretrained()                       │
│  │ ~829K trainable        │                                                                          │
│  │ ~299M frozen (0.28%)   │                                                                          │
│  └────────────────────────┘                                                                          │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
                                          │
┌─────────────────────────────────────────┼────────────────────────────────────────────────────────────┐
│ SECTION B: TRAINING                     │ SECTION C: INFERENCE                                       │
│                                         │                                                            │
│  [RGB+aug] ──→ ┌─────────────┐          │  [Depth .npy] ──→ [Sobel] ──→ [τ=0.03] ──→ [CC] ──→ [Out] │
│                │  STUDENT    │          │                    dx,dy      threshold    min area   inst │
│  [RGB] ──────→ │  (green)    │          │                                          PQ=20.20%       │
│                │  ~829K      │          │                                                            │
│                └──────┬──────┘          │                                                            │
│                       │                 │                                                            │
│              ┌────────▼────────┐        │                                                            │
│              │  LOSS BOX       │        │                                                            │
│              │  ① L_mse: 1.0   │        │                                                            │
│              │  ② L_rank: 0.1  │        │                                                            │
│              │  ③ L_si: 0.5    │        │                                                            │
│              │  TOTAL = sum    │        │                                                            │
│              └────────┬────────┘        │                                                            │
│                       │                 │                                                            │
│              ┌────────▼────────┐        │                                                            │
│              │  TEACHER        │        │                                                            │
│              │  (orange)       │        │                                                            │
│              │  0 params       │        │                                                            │
│              └─────────────────┘        │                                                            │
│                                         │                                                            │
└────────────────────────────────────────────────────────────────────────────────────────────────────┘
```
```