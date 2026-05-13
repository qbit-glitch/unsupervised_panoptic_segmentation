# Image Generation Prompt: MBPS Stage 1 — DINOv2+CAUSE-TR with DoRA Adapters

## Metadata
- **Architecture**: DINOv2 ViT-B/14 + CAUSE-TR + DoRA Adapters
- **Stage**: Stage 1 (Semantic Pseudo-Label Generation)
- **Training Paradigm**: Self-supervised Student-Teacher Distillation
- **Generated**: 2026-04-24
- **Target Venue**: NeurIPS/ICML/CVPR

---

## Prompt

Create a professional ML research paper architectural diagram showing the MBPS Stage 1 adapter-based self-supervised training pipeline for semantic pseudo-label generation. Clean, minimal, publication-quality style similar to NeurIPS/ICML paper figures. White background, no clutter, flat design.

=== LAYOUT ===
Three horizontal sections stacked vertically with thin separator lines:
- Top section (light gray #F5F5F5 background): "ADAPTER INJECTION STRATEGY" (purple accent #7B1FA2)
- Middle section (white background): "STUDENT-TEACHER DISTILLATION LOOP" (blue accent #1565C0 for student, orange accent #E65100 for teacher)
- Bottom section (light gray #F5F5F5 background): "MULTI-OBJECTIVE LOSS & EMA UPDATE" (green accent #2E7D32)
Overall aspect ratio: wide landscape (roughly 2:1)

=== TOP SECTION: ADAPTER INJECTION STRATEGY ===
Left-to-right flow showing the DINOv2 ViT-B/14 backbone with DoRA adapters injected:

[1] INPUT BOX (far left):
    - Label: "DINOv2 ViT-B/14"
    - Subtext: "86M params, 12 blocks, 768-dim"
    - Small icon: frozen snowflake ❄️ on top-right corner

[2] ARROW → into a tall vertical visualization:

[3] TIERED BLOCK VISUALIZATION (center-left, tall vertical stack):
    Show 12 rectangular blocks stacked vertically, each representing a ViT block:
    - Blocks 0–5 (Early Tier): Light blue fill (#E3F2FD)
      Each block internally shows 4 horizontal layers:
        • "attn.qkv" with small green badge "DoRA" (trainable)
        • "attn.proj" with gray badge "frozen"
        • "mlp.fc1" with gray badge "frozen"
        • "mlp.fc2" with gray badge "frozen"
      Small label on left: "Early: qkv-only steering"

    - Blocks 6–11 (Late Tier): Darker blue fill (#1565C0, white text)
      Each block internally shows 4 horizontal layers:
        • "attn.qkv" with green badge "DoRA"
        • "attn.proj" with green badge "DoRA"
        • "mlp.fc1" with green badge "DoRA"
        • "mlp.fc2" with green badge "DoRA"
      Small label on left: "Late: full adaptation"

    - Between blocks 5 and 6: A small dashed line with label "late_block_start=6"

    - Right of the block stack: A vertical annotation bar showing:
      "Trainable: ~424K" in green
      "Frozen: ~86M" in gray

[4] ARROW → right from block stack into:

[5] CAUSE-TR HEAD BOX (center):
    - White rounded rectangle with thin purple border (#7B1FA2)
    - Label: "CAUSE-TR Segment_TR"
    - Subtext: "TRDecoder, 90D reduction, ~2M params"
    - Inside: four small rows with optional green "DoRA" badges on:
      • self_attn.out_proj
      • multihead_attn.out_proj
      • linear1 (FFN expand)
      • linear2 (FFN project)
    - Below: small text "EMA head: frozen" with snowflake icon

[6] ARROW → right into:

[7] OUTPUT BOX (far right):
    - Label: "90-D Segmentation Codes"
    - Subtext: "529 tokens × 90-dim"
    - Below: "→ K-Means (k=54) → Pseudo-Labels"

=== MIDDLE SECTION: STUDENT-TEACHER DISTILLATION LOOP ===
Show two parallel columns with a central loss block below.

LEFT COLUMN — STUDENT (blue accent #1565C0):
[1] INPUT IMAGE (top):
    - Small Cityscapes street scene thumbnail
    - Label: "Input Image (322×322)"
    - Small "Aug 1 (weak)" badge

[2] ARROW down into:

[3] STUDENT MODEL BOX:
    - Blue-tinted layer-slab visualization (3D rectangular slabs, progressively narrower then wider)
    - Label: "STUDENT" in bold
    - Subtext: "DINOv2 + DoRA adapters"
    - Green indicator dot "trainable"
    - Right side: small parameter count "472K trainable"

[4] ARROW down into:

[5] STUDENT FEATURE OUTPUT:
    - Small grid visualization: 23×23 patch tokens
    - Label: "feat_student [529×768]"
    - ARROW right → into CAUSE-TR head (small blue box)
    - Output: "seg_student [529×90]"

RIGHT COLUMN — TEACHER (orange accent #E65100):
[1] INPUT IMAGE (top):
    - Same Cityscapes image with dashed border indicating shared input
    - Label: "Input Image"
    - Small "Aug 2 (strong)" badge with "optional" note

[2] ARROW down into:

[3] TEACHER MODEL BOX:
    - Orange-tinted layer-slab visualization
    - Label: "TEACHER" in bold
    - Subtext: "DINOv2 (original W₀)"
    - Snowflake/lock icon "frozen"
    - Right side: "86M frozen"

[4] ARROW down into:

[5] TEACHER FEATURE OUTPUT:
    - Small grid visualization: 23×23 patch tokens
    - Label: "feat_teacher [529×768]"
    - ARROW right → into CAUSE-TR head (small orange box, "frozen")
    - Output: "seg_teacher [529×90]"

CONNECTING ARROWS:
- From feat_student [529×768]: two arrows converge down to loss block
- From feat_teacher [529×768]: two arrows converge down to loss block
- From seg_student [529×90]: arrow converges down to loss block
- From seg_teacher [529×90]: arrow converges down to loss block

=== BOTTOM SECTION: MULTI-OBJECTIVE LOSS & EMA UPDATE ===
Central horizontal layout showing four loss components side-by-side, then EMA update below.

[1] LOSS COMPONENTS (left to right, four rounded rectangles with colored tops):
    BOX 1 (purple top #7B1FA2): "DINO Distillation"
    - Equation: "KL(softmax(f_s/0.1) || softmax(f_t/0.07))"
    - Subtext: "τ_student=0.1, τ_teacher=0.07"

    BOX 2 (teal top #00897B): "Depth Correlation"
    - Equation: "-corr(seg_s, depth_map)"
    - Subtext: "λ_depth = 0.05"

    BOX 3 (indigo top #303F9F): "Cross-View Consistency"
    - Equation: "1 - cos(feat_s, feat_aug)"
    - Subtext: "L2-cosine alignment"

    BOX 4 (deep orange top #E65100): "CAUSE Cluster Loss"
    - Equation: "VQ_loss(head_ema(feat_t))"
    - Subtext: "EMA head commitment"

[2] Below the four boxes, a large bracket "{ }" combining them into:

[3] TOTAL LOSS BOX:
    - Equation in calligraphic font:
      "L_total = w₁·L_distill + w₂·L_crossview + w₃·L_depth + w₄·L_cluster"
    - Arrow down labeled "AdamW (lr=1e-4, wd=1e-4)"

[4] EMA UPDATE BOX (bottom-right, separate):
    - Label: "EMA Head Update"
    - Equation: "head_ema ← 0.99·ema + 0.01·head"
    - Small arrow curving back UP to the teacher column showing the EMA head
    - Dashed arrow style, green color (#2E7D32)

=== SIDE ANNOTATION (right edge, spanning middle+bottom sections) ===
A vertical parameter-efficiency callout box:
┌─────────────────────────┐
│  Parameter Efficiency   │
│  ─────────────────────  │
│  Total: 86M+ params     │
│  Trainable: ~472K       │
│  (0.55% of backbone)    │
│                         │
│  DoRA r=4, α=4.0        │
│  late_block_start=6     │
└─────────────────────────┘

=== STYLE GUIDELINES ===
- Clean white background for middle section, light gray (#F5F5F5) for top/bottom
- Neural network blocks shown as 3D rectangular slabs with perspective (like looking at a bookshelf from an angle)
- Early tier blocks: light blue (#E3F2FD); Late tier blocks: dark blue (#1565C0) with white text
- DoRA badges: small green (#4CAF50) rounded pills with white text
- Frozen badges: small gray (#9E9E9E) rounded pills
- Student column: blue accent (#1565C0); Teacher column: orange accent (#E65100)
- Loss boxes: white fill with colored top bar (4px height)
- Thin black arrows with solid arrowheads connecting stages
- Curved dashed green arrow for EMA feedback loop
- Real Cityscapes thumbnail at input (urban driving scene)
- Sans-serif typography, consistent sizing hierarchy
- Flat design: no gradients, no drop shadows on boxes, subtle 1px borders
- Equations in proper mathematical notation (LaTeX-style subscripts, calligraphic L)
- Minimal text: 2-4 words per label, equations where possible

=== REFERENCE ASCII LAYOUT ===

┌─ ADAPTER INJECTION STRATEGY ─────────────────────────────────────────────┐
│                                                                          │
│  DINOv2-B/14 ──→ ┌─ Blocks 0-5 ─┐  ┌─ Blocks 6-11 ─┐ ──→ CAUSE-TR ──→  │
│   86M frozen      │ qkv◄DoRA     │  │ qkv◄DoRA      │      +DoRA opt    │
│                   │ proj frozen  │  │ proj◄DoRA     │      90D codes    │
│                   │ mlp frozen   │  │ mlp◄DoRA      │                   │
│                   └──────────────┘  └───────────────┘     ~472K train   │
│                        ▲    late_block_start=6                           │
└────────────────────────┼─────────────────────────────────────────────────┘
                         │
┌─ STUDENT-TEACHER LOOP ─┼─────────────────────────────────────────────────┐
│                        │                                                 │
│   ┌──────────┐    ┌────▼────┐              ┌──────────┐    ┌────────┐  │
│   │ Input    │───→│ STUDENT │              │ TEACHER  │    │ Input  │  │
│   │ (weak)   │    │ +DoRA   │              │ W₀ frozen│    │(strong)│  │
│   └────┬─────┘    └────┬────┘              └────┬─────┘    └───┬────┘  │
│        │               │                        │               │       │
│   feat_student      seg_student            feat_teacher      seg_teacher │
│   [529×768]         [529×90]               [529×768]        [529×90]    │
│        │               │                        │               │       │
│        └───────────────┼────────────────────────┘               │       │
│                        ▼                                        │       │
└────────────────────────┼────────────────────────────────────────┘       │
                         │                                                │
┌─ LOSS & EMA UPDATE ────┼────────────────────────────────────────────────┘
│                        ▼
│   ┌─────────┐ ┌───────────┐ ┌───────────────┐ ┌─────────────┐
│   │DINO KL  │ │Depth Corr │ │Cross-View L2  │ │VQ Cluster   │
│   │τ=0.1/0.7│ │λ=0.05     │ │Cosine Align   │ │EMA Commit   │
│   └────┬────┘ └─────┬─────┘ └───────┬───────┘ └──────┬──────┘
│        └─────────────┴───────────────┴────────────────┘
│                          │
│                    L_total = Σ wᵢ·Lᵢ
│                          │
│                          ▼
│                   AdamW step (adapters only)
│                          │
│                ┌─────────┴─────────┐
│                │ head_ema ← 0.99·  │
│                │   ema + 0.01·head │
│                └───────────────────┘
│                          │
│                (curved dashed arrow back up to teacher)
└─────────────────────────────────────────────────────────────────────────┘
