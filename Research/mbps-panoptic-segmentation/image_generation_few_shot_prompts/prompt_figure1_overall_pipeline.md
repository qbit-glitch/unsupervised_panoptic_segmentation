## Prompt: Figure 1 Overall Pipeline Overview

```md
Create a clean CVPR/NeurIPS-style overview diagram for our method. The figure
should be short, precise, and architectural, not detailed. It should resemble a
compact paper pipeline figure: one horizontal flow, minimal text, clear arrows,
and small visual thumbnails.

=== GOAL ===
Show the full method at a glance:

RGB image -> frozen monocular priors -> DCFA/SIMCF pseudo-label generation
-> panoptic pseudo-labels -> CUPS-style panoptic bootstrapping
-> EMA self-training -> final panoptic output.

Do not show low-level details, losses, hyperparameters, equations, ablation
numbers, or implementation internals.

=== LAYOUT ===
Single horizontal pipeline, left to right, wide and shallow aspect ratio
similar to a paper teaser pipeline, roughly 5:1.

Use two lightly labeled regions:

1. Left region title:
   "Pseudo-Label Generation"

2. Right region title:
   "Unsupervised Panoptic Training"

Place the title text above each region. Use a thin dashed vertical divider
between the two regions.

=== PIPELINE ELEMENTS ===

[1] INPUT
    Small Cityscapes-style RGB street-scene thumbnail.
    Label: "RGB Image"

Arrow to:

[2] FROZEN PRIORS
    Draw two small stacked boxes or icons:
      top: "Semantic Prior"
      bottom: "Depth Prior"
    Add tiny lock/snowflake icons to show both are frozen.
    Keep text minimal.

Arrow to:

[3] DCFA + SIMCF
    One compact processing box.
    Label: "DCFA + SIMCF"
    Small subtitle: "Cross-modal agreement"
    Visual hint: semantic colors + depth contour lines merging together.

Arrow to:

[4] PANOPTIC PSEUDO-LABEL
    Small colorful panoptic segmentation thumbnail.
    Label: "Panoptic Pseudo-Label"

Long arrow crossing the dashed divider to:

[5] PANOPTIC BOOTSTRAPPING
    Compact box.
    Label: "Bootstrapping"
    Small subtitle: "DropLoss + CopyPaste"

Arrow to:

[6] PANOPTIC NETWORK TRAINING
    Draw a simple network block labeled:
      "Panoptic Network"
    Add a small circular arrow around it labeled:
      "EMA self-training"
    Keep the loop subtle and clean.

Arrow to:

[7] OUTPUT
    Small final panoptic prediction thumbnail.
    Label: "Output Panoptic Map"

=== VISUAL STYLE ===
- White background.
- Flat design.
- Thin black arrows with clear arrowheads.
- Minimal text, 2-4 words per label where possible.
- Use real-looking Cityscapes thumbnails or simple thumbnail placeholders.
- Use semantic maps with class colors and instance maps with distinct object colors.
- Frozen priors should have small lock/snowflake icons.
- Use blue accents for pseudo-label generation.
- Use purple/orange accents for training.
- No tables, no metrics, no formulas, no long descriptions.
- Do not include stereo, optical flow, lidar, GT labels, or Mamba blocks.
- The figure should communicate the architectural pipeline in one glance.

=== REFERENCE ASCII LAYOUT ===

        Pseudo-Label Generation                         Unsupervised Panoptic Training

  [RGB Image] -> [Frozen Priors] -> [DCFA + SIMCF] -> [Panoptic PL]  |  -> [Bootstrapping] -> [Panoptic Network] -> [Output]
                   Semantic 🔒        Cross-modal        colorful    |      DropLoss          EMA loop              panoptic
                   Depth 🔒           agreement          pseudo-label |      CopyPaste         self-training         prediction

Compact visual version:

┌────────────── Pseudo-Label Generation ──────────────┐  ┌──── Unsupervised Panoptic Training ────┐
│                                                     │  │                                        │
│ [RGB] -> [Semantic Prior 🔒]                        │  │                                        │
│          [Depth Prior 🔒] -> [DCFA+SIMCF] -> [PL] --┼->│ [Bootstrapping] -> [Panoptic Network] -> [Output]
│                          cross-modal agreement      │  │  DropLoss+CP        EMA self-train     │
│                                                     │  │                                        │
└─────────────────────────────────────────────────────┘  └────────────────────────────────────────┘
```
