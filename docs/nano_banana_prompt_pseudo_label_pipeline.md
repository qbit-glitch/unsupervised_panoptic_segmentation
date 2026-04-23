# Nano Banana Prompt: Research-Grade Paper Figure
## MBPS Pseudo-Label Generation Pipeline (k=80 + DepthPro + DCFA + SIMCF-ABC)

> **Author-verified:** No DINOv3 features are used in pseudo-label generation. DINOv3 appears only as the frozen downstream training backbone. Semantic features come from DINOv2 (CAUSE). Instance generation uses DepthPro only. SIMCF-ABC uses only labels + depth maps.

---

### Prompt Text (copy-paste into Nano Banana)

```
A clean, publication-quality vector-style architecture diagram for a computer vision research paper (CVPR/ICCV/NeurIPS quality). White background. Professional, minimal, with sharp lines and modern sans-serif labels.

LAYOUT: Horizontal left-to-right pipeline flow with 4 color-coded stages. Each stage has a subtle colored banner at top:
- Stage 1 (Feature Level): Deep blue banner
- Stage 2 (Instance Level): Emerald green banner  
- Stage 3 (Label Level): Amber/orange banner
- Stage 4 (Panoptic Merge): Purple banner

LEFT SIDE — INPUTS (2 stacked boxes):
[Box 1] "Cityscapes RGB" with a small inset photo of a street scene (cars, road, buildings, sky). Label below: "1024×2048"
[Box 2] "DepthPro Depth" with a grayscale depth map visualization of the same scene (darker = farther). Label: "512×1024, [0,1]"

STAGE 1 — SEMANTIC PSEUDO-LABEL (blue banner):
A rounded rectangle labeled "CAUSE Segment_TR (DINOv2 ViT-B/14)" outputting "90-D codes". Small annotation: "NOT DINOv3".

These 90-D codes feed into a central rounded rectangle labeled "DCFA Depth-Conditioned Feature Adapter". Inside:
  - Top-left small box: "90-D CAUSE Codes (DINOv2)"
  - Top-right small box: "16-D Sinusoidal Depth" with tiny sine-wave icons
  - Both feed into a vertical MLP diagram: 2 stacked rectangles labeled "Linear(106→384) + LayerNorm + ReLU" and "Linear(384→384) + LayerNorm + ReLU", then "Linear(384→90) [ZERO-INIT]"
  - A curved arrow bypassing the MLP labeled "Skip Connection: adjusted = codes + residual"
  - Bottom annotation: "~40K params | λ_preserve = 20.0"

Arrow right to a cylinder icon labeled "MiniBatchKMeans" with "k=80" prominently displayed. Below: "batch=10K | max_iter=300 | n_init=3"

Arrow right to a grid labeled "80 Cluster IDs" (visualized as a small 32×64 heatmap with 80 discrete colors). Arrow down to "Cluster→Class Mapping" (small table icon with 80 rows → 19 classes via majority vote).

Arrow right to output: a semantic segmentation map (colored street scene) labeled "Semantic Pseudo-Label | 19 classes | PQ_stuff 33.96"

STAGE 2 — INSTANCE PSEUDO-LABEL (green banner):
A clear annotation at top: "NO vision features used — DepthPro only"

Input from DepthPro depth map branches down to this stage.
A processing chain:
  [Box] "Gaussian σ=0" → [Box] "Sobel ∇D" with small kernel icon (Gx, Gy) → [Box] "||∇D|| > τ=0.20" with a threshold gate symbol
  → [Box] "Per-Class Connected Components" showing 8 small thing-class icons (person, car, truck, bus, bicycle, motorcycle, rider, train)
  → [Filter icon] "Area ≥ 1000 px"
  → [Dilation icon] "Dilate 3×"

Output: colored instance masks overlaid on the street scene, each instance a different bright color. Label: "~17 instances/image | PQ_things 14.70"

STAGE 3 — SIMCF-ABC REFINEMENT (amber banner):
Three stacked processing blocks with left-to-right arrows:

Block A — "Instance Validates Semantics":
  Small diagram showing an instance mask (blob) with internal pixels being recolored to match majority class. Label: "Majority Vote | Step A"

Block B — "Semantics Validate Instances" (most prominent, slightly larger):
  Two adjacent instance blobs with a merge arrow between them. Clear label: "NO DINOv2/DINOv3 features used". Below: "Union-Find Merge | 44→22 inst/image | +1.33 PQ_things". The merge decision is based on adjacency + same semantic class only.

Block C — "Depth Validates Semantics":
  A Gaussian bell curve with shaded tails beyond ±3σ. Pixels in the tail regions are masked with a crosshatch pattern. Label: "3-Sigma Outlier Mask | ~85M px (1.36%) | +0.30 PQ_stuff"

STAGE 4 — PANOPTIC MERGE (purple banner):
Two inputs converge: refined semantic map + refined instance masks.
A vertical 3-step flowchart:
  Step 1 (top): "Place Things First" — thing instances (cars, people) placed on canvas with priority
  Step 2 (middle): "Fill Stuff" — road, sky, vegetation fill remaining gaps
  Step 3 (bottom): "Fallback CC" — small connected-component fragments for uncovered thing pixels

Formula prominently displayed in center: "panoptic_id = class_id × 1000 + instance_id"

Output (far right): Final panoptic segmentation map where stuff regions have uniform colors and thing instances each have unique bright colors with thin white borders. Large label: "Panoptic Pseudo-Label | PQ = 25.85"

BOTTOM RIGHT CORNER:
A small results table (publication-style) showing ablation:
  Baseline PQ=24.54 | +DCFA PQ=25.22 | +SIMCF PQ=25.27 | FULL PQ=25.85

FAR BOTTOM (small annotation):
"Stage-2 Training Backbone: DINOv3 ViT-B/16 (frozen) — downstream only, NOT used in pseudo-label generation"

OVERALL STYLE NOTES:
- All boxes have 2px rounded corners with subtle drop shadows
- Arrows are thick (3px), dark gray, with triangular heads
- Tensor dimensions shown in small monospace font next to data arrows
- Color palette: professional, not neon — use muted blues, greens, ambers, purples
- No 3D effects, no gradients, flat design
- Conference figure caption area at bottom: "Figure 2: Overview of our depth-conditioned pseudo-label generation pipeline. DCFA adapts DINOv2-based CAUSE features using DepthPro geometry, k=80 overclustering produces fine-grained semantics, depth-guided connected components extract instances without any vision features, and SIMCF-ABC enforces cross-modal consistency using only labels and depth maps."
```

---

### Alternative / Condensed Prompt (if token limit is tight)

```
Publication-quality architecture diagram, white background, flat vector style, CVPR paper figure. Horizontal 4-stage pipeline left-to-right.

Stage 1 (blue banner): "CAUSE DINOv2 ViT-B/14" box outputting "90-D codes" with small "NOT DINOv3" annotation. These feed "DCFA" box with 2-layer MLP inside, inputs "90D codes + 16D sinusoidal depth", skip connection arrow, output to "k=80 K-Means" cylinder, then "cluster→class mapping" table, output colored semantic segmentation map of a street scene.

Stage 2 (green banner): Clear label "DepthPro ONLY — no vision features". "DepthPro Depth" feeds "Sobel edges τ=0.20" gate, then "per-class connected components" for 8 thing classes, "area≥1000px" filter, "dilate 3×", output instance masks overlay on scene.

Stage 3 (amber banner): Three stacked blocks — A) "majority vote consistency", B) "merge adjacent instances" with union-find icon (highlight this block as most important; add clear label "NO DINOv2/DINOv3 features — adjacency + class only"), C) "3-sigma depth outlier mask" with Gaussian curve.

Stage 4 (purple banner): Semantic + instance inputs merge with formula "panoptic_id = class_id × 1000 + instance_id". Three-step vertical flow: "place things first", "fill stuff", "fallback CC". Output: panoptic segmentation map with stuff in uniform colors and things in bright bordered instance colors.

Bottom annotation: "Training backbone: DINOv3 ViT-B/16 (frozen) — downstream only". Inset small ablation table bottom-right: Baseline 24.54 → Full 25.85 PQ. Clean labels, 3px arrows, rounded boxes, no 3D, professional muted colors.
```

---

### Negative Prompt (add to suppress unwanted artifacts)

```
No 3D renders, no photorealistic style, no blurry text, no watercolor, no sketch, no hand-drawn look, no cartoon style, no neon colors, no cluttered background, no shadows on text, no perspective distortion, no photographic elements except the small inset street scene, no gradient fills on boxes, no rounded arrowheads.
```

---

## Recommended Nano Banana Settings

| Setting | Recommendation |
|---------|---------------|
| **Aspect Ratio** | 16:9 or 21:9 (wide landscape for horizontal pipeline) |
| **Style** | "Technical diagram" / "Vector illustration" / "Infographic" |
| **Quality** | Max quality / High detail |
| **Steps** | 50+ |
| **CFG Scale** | 7.0–8.5 (balance adherence vs creativity) |

---

*Prompt designed for MBPS pseudo-label pipeline figure — DCFA + k=80 + DepthPro + SIMCF-ABC. Author-verified: no DINOv3 features in pseudo-label generation.*
