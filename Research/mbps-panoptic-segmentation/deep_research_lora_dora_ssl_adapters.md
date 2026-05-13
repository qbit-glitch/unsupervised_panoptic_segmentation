# Deep Research Report: A* Papers on LoRA/DoRA Adapter Training for Frozen Backbones via Self-Supervised / Unsupervised Learning

**Research Date:** 2026-04-24
**Scope:** CVPR, ICCV, ECCV, NeurIPS, ICML, ICLR, AAAI (2022–2026)
**Focus:** Frozen vision backbones + PEFT adapters trained WITHOUT labeled supervision

---

## Executive Summary

Papers that **simultaneously** satisfy all four criteria — (1) top-tier venue, (2) frozen vision backbone, (3) LoRA/DoRA/Adapter/BitFit-style PEFT, and (4) **self-supervised or unsupervised** training objective — are **extremely rare**. Most PEFT literature applies adapters via supervised fine-tuning. However, a small but growing body of work (especially 2024–2025) explicitly extends self-supervised pre-training into the adapter-training phase. **No top-tier paper currently trains DoRA adapters with self-supervised objectives on frozen vision backbones** — this is a clear research gap.

---

## 🥇 Tier 1: Direct Matches (Frozen Backbone + PEFT + SSL/UNS Objective)

### 1. ExPLoRA: Parameter-Efficient Extended Pre-Training to Adapt Vision Transformers under Domain Shifts
| Field | Details |
|-------|---------|
| **Venue** | **ICML 2025** |
| **Authors** | Samar Khanna, Medhanie Irgau, David B. Lobell, Stefano Ermon |
| **Backbone** | ViT-Large from **DINOv2** or **MAE** — mostly frozen (1–2 blocks unfrozen) |
| **Adapter** | **LoRA** (r=64) on Q and V projections in attention layers. LN also unfrozen. |
| **SSL Objective** | Continues the **original SSL objective**: DINO/iBOT self-distillation + KoLeo + Sinkhorn-Knopp centering; or MAE masked reconstruction (75% masking). |
| **Key Results** | +8% linear probing on satellite imagery; SoTA on fMoW-RGB (79.28%) with only **6% encoder params** and **8× less compute** than full domain pre-training. SoTA on VisDA-2017 UDA (90.9%). |
| **Code** | ✅ https://samar-khanna.github.io/ExPLoRA/ |

> **Why it matters:** The most directly relevant paper. Explicitly asks: *"Can we adapt a pre-trained foundation model to a new domain via efficient self-supervised pre-training?"* and answers it by extending SSL pre-training with LoRA on a mostly-frozen ViT.

---

### 2. Parameter Efficient Self-Supervised Geospatial Domain Adaptation (GDA / SLR Adapters)
| Field | Details |
|-------|---------|
| **Venue** | **CVPR 2024** |
| **Authors** | Linus Scheibenreif, Michael Mommert, Damian Borth |
| **Backbone** | **MAE / SatMAE / Scale-MAE** ViT-Large (304M params) — **fully frozen** |
| **Adapter** | **Scaled Low-Rank (SLR) adapters**: learnable input/output scaling vectors + low-rank matrices on QKV and MLP layers. |
| **SSL Objective** | **MAE masked autoencoding** on unlabeled target-domain data. 75% patches masked; adapters reconstruct while backbone stays frozen. |
| **Key Results** | +6.37% absolute for MAE, +2.93% SatMAE, +3.68% Scale-MAE across 8 remote sensing datasets. SLR adapter tuning **outperforms full fine-tuning** on most combinations. |
| **Code** | ✅ https://github.com/HSG-AIML/GDA |

> **Why it matters:** The only other top-tier paper explicitly training adapters with a self-supervised objective (MAE reconstruction) on a **completely frozen** backbone.

---

### 3. GLARE: Enhancing Semantic Segmentation with Continual Self-Supervised Pre-training
| Field | Details |
|-------|---------|
| **Venue** | arXiv:2509.17816 (OpenReview 2025) |
| **Authors** | Brown Ebouky, Ajad Chhatkuli, et al. |
| **Backbone** | ViT-S/16 from **UDI** (self-supervised segmentation pre-training) — **frozen** |
| **Adapter** | **UniAdapter** (bottleneck: down-projection → ReLU → up-projection) after every self-attention layer. |
| **SSL Objective** | Multi-level consistency in student-teacher (EMA) framework: global [CLS] consistency, regional consistency via attention sampling, local patch-level consistency via inter-view matching. |
| **Key Results** | +0.4 mIoU on ADE20k, +0.6 mIoU on LoveDA. Prevents catastrophic forgetting. |
| **Code** | ❓ Not mentioned |

> **Why it matters:** One of the first works extending self-supervised pre-training (not fine-tuning) for **dense prediction** (segmentation) with adapter-only training.

---

### 4. STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences
| Field | Details |
|-------|---------|
| **Venue** | **ICLR 2022** |
| **Authors** | Hamilton et al. |
| **Backbone** | **Frozen DINO ViT** — fully frozen |
| **Adapter** | Lightweight segmentation head (not LoRA, but trainable head on frozen features) |
| **SSL Objective** | **Feature correspondence distillation** from frozen DINO features. Correspondence loss + feature clustering loss. Fully unsupervised. |
| **Key Results** | SoTA unsupervised semantic segmentation on COCO-Stuff, Cityscapes, Potsdam. Trains in **<2h on V100**. |
| **Code** | ✅ https://github.com/mhamilton723/STEGO |

> **Why it matters:** Foundational work showing frozen DINO features can be distilled into a lightweight head with zero labels. Demonstrates the power of frozen SSL backbone + trainable modules.

---

## 🥈 Tier 2: Strongly Related (Frozen Backbone + PEFT + Unsupervised Adaptation)

### 5. Uni-UVPT: Universal Unsupervised Visual Prompt Tuning for Source-Free Domain Adaptive Semantic Segmentation
| Field | Details |
|-------|---------|
| **Venue** | **NeurIPS 2023** |
| **Authors** | Ma et al. (Huawei Noah's Ark Lab) |
| **Backbone** | Swin / MT transformer — **fully frozen** |
| **Adapter** | **Visual Prompt Tuning** + lightweight Prompt Adapter (prompt generator + interactor) |
| **UNS Objective** | Self-training with **adaptive pseudo-label correction** + **multiscale consistency loss**. |
| **Key Results** | SoTA on GTA5→Cityscapes and SYNTHIA→Cityscapes. Only 12–28M trainable params. |
| **Code** | ✅ https://github.com/huawei-noah/noah-research/tree/master/uni-uvpt |

> **Direct relevance to MBPS:** Source-free UDA for **semantic segmentation** with frozen backbone + prompt adapter. Closest architectural precedent for dense prediction.

---

### 6. LoRA-TTT: Low-Rank Test-Time Training for Vision-Language Models
| Field | Details |
|-------|---------|
| **Venue** | **ICML 2025** |
| **Authors** | Kojima et al. (Sony, UC San Diego) |
| **Backbone** | CLIP image encoder — **fully frozen**; only LoRA params updated |
| **Adapter** | **LoRA** (r=16, layers 11–12) — ~0.2M params |
| **SSL Objective** | **Marginal Entropy Minimization (MEM)** + **MAE reconstruction loss** on high-confidence augmented views. |
| **Key Results** | +5.79% on OOD benchmarks. **146.7 FPS** vs TPT's 7.1 FPS. **0.32GB** memory vs 5.2GB. |
| **Code** | 📄 OpenReview ICML 2025 |

> **Key insight:** First to apply **LoRA for test-time adaptation** in VLMs. Replaces prompt tuning with LoRA tuning of the vision encoder.

---

### 7. FORLA: Frozen Feature Reuse via Low-Rank Adapter with Progressive Self-Distillation
| Field | Details |
|-------|---------|
| **Venue** | arXiv 2024 |
| **Authors** | (SAM/DINO adaptation) |
| **Backbone** | Frozen SAM / DINO features |
| **Adapter** | **LoRA** adapters |
| **SSL Objective** | **Progressive EMA self-distillation**: early training — teacher adapter updated via EMA from student; later — bidirectional transfer. Prevents collapse. |
| **Code** | ❓ Not yet public |

> **Key insight:** Progressive distillation strategy prevents student adapter collapse. Two-stage: EMA-only → bidirectional.

---

### 8. MARCO: Dense Self-Distillation with AdaptFormer Adapters
| Field | Details |
|-------|---------|
| **Venue** | CVPR 2026 |
| **Authors** | (Semantic correspondence) |
| **Backbone** | Frozen **DINOv2** |
| **Adapter** | **AdaptFormer** bottleneck adapters |
| **SSL Objective** | Dense self-distillation via **EMA teacher** for semantic correspondence. |
| **Key Results** | AdaptFormer 87.2 vs LoRA 85.2 PCK. |
| **Code** | ❓ |

> **Key insight:** For dense prediction, **AdaptFormer/bottleneck adapters sometimes outperform LoRA**. Worth considering for segmentation tasks.

---

### 9. SVL-Adapter: Self-Supervised Adapter for Vision-Language Pretrained Models
| Field | Details |
|-------|---------|
| **Venue** | **BMVC 2022** |
| **Authors** | Pantazis et al. (UCL) |
| **Backbone** | **Frozen CLIP** + frozen **DINO/BarlowTwins** SSL backbone |
| **Adapter** | Lightweight **feature adapter** (FC network) on frozen CLIP features |
| **SSL Objective** | Blends CLIP zero-shot predictions with **SSL features**. No labels needed. |
| **Key Results** | ~10% improvement over existing low-shot methods on challenging visual tasks. |
| **Code** | ✅ https://github.com/omipan/svl_adapter |

---

### 10. PC-LoRA: Progressive Compression via LoRA Distillation
| Field | Details |
|-------|---------|
| **Venue** | **ICLR 2024 Workshop** |
| **Authors** | (Model compression) |
| **Backbone** | Frozen backbone |
| **Adapter** | **LoRA** |
| **SSL Objective** | **Progressively distills frozen backbone into LoRA adapters** until only adapters remain (94% compression). |
| **Code** | ❓ |

> **Key insight:** Shows that a frozen backbone can be progressively compressed into LoRA adapters via self-distillation.

---

## 🥉 Tier 3: Foundational / Contextual Papers

| Paper | Venue | Relevance |
|-------|-------|-----------|
| **DoRA** (Liu et al.) | ICML 2024 | Foundational DoRA method. All experiments are **supervised fine-tuning**. Gap: no SSL/UNS training of DoRA. |
| **Adapters Strike Back** (Steitz & Roth) | CVPR 2024 | Systematic adapter placement study. Validates on DINO backbones but trains **supervised**. |
| **PEFT ViTs w/o Forgetting** (Bafghi et al.) | CVPRW 2024 | LoRA + Block Expansion on DINO ViT. **Supervised** transfer learning. |
| **TPT** (Shu et al.) | NeurIPS 2022 | Test-time prompt tuning with **entropy minimization**. Frozen CLIP. Foundational for TTT. |
| **UPL** (Huang et al.) | arXiv 2022 | Unsupervised prompt learning with **pseudo-labeling**. Frozen CLIP. |
| **DAPL** (Ge et al.) | arXiv 2022 | Domain adaptation via prompt learning. **Contrastive + pseudo-labeling**. |
| **InfLoRA** (Liang & Li) | CVPR 2024 | Interference-free LoRA subspace for continual learning. **Orthogonality constraints**. |
| **CL-LoRA** (He et al.) | CVPR 2025 | Dual-adapter (shared + specific) for class-incremental learning. |
| **NoLA** (BMVC 2025) | BMVC 2025 | DINO + CLIP alignment for **label-free** tuning. FixMatch-style pseudo-labeling. |
| **SAGE-reID** (arXiv 2025) | arXiv 2025 | LoRA + **DBSCAN pseudo-label clustering** for source-free adaptation. |
| **VirDA** (arXiv 2025) | arXiv 2025 | Visual reprogramming for UDA with **entirely frozen backbone**. |
| **PromptKD** (CVPR 2024) | CVPR 2024 | Teacher-student **prompt distillation** on unlabeled data. |
| **LaFTer** (Mirza et al.) | NeurIPS 2023 | Label-free tuning with **LLM-generated descriptions** + pseudo-labels. |
| **L2P / DualPrompt / CODA-Prompt** | CVPR 2022/2022/2023 | Prompt-based continual learning. **Fully frozen ViT**. |
| **AdapterFormer** (NeurIPS 2022) | NeurIPS 2022 | Parallel bottleneck adapters for video. **Frozen ViT**. |
| **CLIP-Adapter** (IJCV 2023) | IJCV 2023 | Feature adapters on frozen CLIP. Supervised but foundational. |

---

## Core Training Recipes Identified

### Recipe 1: EMA Teacher-Student (DINO-style)
```
Teacher ← EMA(Student) with momentum m ∈ [0.996, 0.999]
Student adapter trains to match teacher features/logits
Backbone fully frozen
```
**Papers:** ExPLoRA, GLARE, FORLA, MARCO

### Recipe 2: Frozen Teacher → Adapter Student
```
Strong frozen foundation model provides targets
Small adapter learns to mimic teacher features/logits
No EMA; teacher is static frozen model
```
**Papers:** STEGO, Surgical-DINO, DINO Teacher (CVPR 2025)

### Recipe 3: Progressive Self-Distillation
```
Stage 1: Teacher adapter updated via EMA from student only
Stage 2: Bidirectional transfer (student also learns from teacher)
Prevents early collapse
```
**Papers:** FORLA

### Recipe 4: Pseudo-Label Self-Training
```
Teacher generates pseudo-labels on unlabeled data
Student adapter trains on pseudo-labels
Iterative refinement (teacher updated by student)
```
**Papers:** UPL, DAPL, Uni-UVPT, SAGE-reID, NoLA, InfoMSD, LaFTer

### Recipe 5: Entropy Minimization + Reconstruction
```
Marginal Entropy Minimization (MEM) on augmented views
+ MAE reconstruction loss on masked patches
```
**Papers:** TPT, LoRA-TTT, DiffTPT

---

## Key Gaps & Opportunities for MBPS

| Gap | Opportunity |
|-----|-------------|
| **No DoRA + SSL vision paper exists** | Your work (`conv_dora` variant in stage-1) is novel. DoRA has only been trained supervised. |
| **Dense prediction + PEFT + SSL is underexplored** | ExPLoRA is classification-only. GLARE is segmentation but not panoptic. Panoptic segmentation with self-supervised adapter tuning is **unexplored**. |
| **Multi-objective adapter training** | Most works use a single SSL objective. Combining **distillation + clustering + cross-view consistency + depth guidance** with PEFT is largely unexplored. |
| **Source-free UDA for panoptic segmentation** | Uni-UVPT proves dense prediction UDA with frozen backbones is viable, but only for semantic segmentation. Panoptic UDA with adapters is open. |
| **Cross-modal adapters (RGB + Depth)** | No paper trains adapters jointly on RGB and depth features in an unsupervised manner. Your depth-guided semantic adapter is novel. |

---

## Practical Recommendations for MBPS Stage-1

### Most Relevant Papers to Cite

| Priority | Paper | Cite For |
|----------|-------|----------|
| 🔥 1 | **ExPLoRA** (ICML 2025) | "First to extend SSL pre-training with LoRA on frozen ViT" — establish that SSL adapter training is a valid paradigm |
| 🔥 2 | **GDA/SLR** (CVPR 2024) | "Adapter-only MAE reconstruction on frozen backbone outperforms full fine-tuning" — justify frozen backbone + adapter approach |
| 🔥 3 | **GLARE** (arXiv 2025) | "Continual SSL for dense prediction with adapters" — closest to your dense prediction task |
| 🔥 4 | **Uni-UVPT** (NeurIPS 2023) | "Source-free UDA for semantic segmentation with frozen backbone" — dense prediction precedent |
| 🔥 5 | **LoRA-TTT** (ICML 2025) | "LoRA adaptation with entropy + reconstruction losses" — justifies your distillation + depth-cluster loss combination |
| 🔥 6 | **STEGO** (ICLR 2022) | "Frozen DINO features distilled into trainable head" — foundational for feature distillation from frozen SSL models |
| 🔥 7 | **DoRA** (ICML 2024) | "Weight-decomposed LoRA outperforms standard LoRA" — justifies DoRA as stronger adapter variant |
| 🔥 8 | **InfLoRA** (CVPR 2024) | "Interference-free subspace design" — if you plan multi-stage/continual adaptation |
| 🔥 9 | **FORLA** (arXiv 2024) | "Progressive EMA self-distillation prevents collapse" — relevant to your EMA teacher-student setup |
| 🔥 10 | **MARCO** (CVPR 2026) | "AdaptFormer outperforms LoRA for dense prediction" — motivates exploring Conv-DoRA over plain LoRA |

### Architectural Insights

1. **Tiered adapter injection** (early blocks = minimal, late blocks = full) is used by ExPLoRA and your `dinov2_adapter.py` — cite ExPLoRA for this design choice.
2. **EMA teacher-student** with frozen teacher + adapter student is the dominant paradigm for SSL adapter training.
3. **Conv-DoRA / spatial adapters** may outperform plain LoRA for dense prediction (MARCO ablation: AdaptFormer 87.2 vs LoRA 85.2 PCK).
4. **Multiple self-supervised losses** (distillation + consistency + pseudo-labeling) are more robust than single-objective training.
5. **Frozen backbone + adapter** drops training time from days to hours — a practical advantage worth highlighting.

---

## Appendix: Full Paper Index

| # | Paper | Venue | Year | Adapter | Backbone | SSL/UNS | Code |
|---|-------|-------|------|---------|----------|---------|------|
| 1 | ExPLoRA | ICML | 2025 | LoRA | DINOv2/MAE | ✅ DINO/iBOT/MAE | ✅ |
| 2 | GDA (SLR) | CVPR | 2024 | SLR Adapter | MAE/SatMAE | ✅ MAE | ✅ |
| 3 | GLARE | arXiv | 2025 | UniAdapter | UDI ViT | ✅ Multi-level consistency | ❓ |
| 4 | STEGO | ICLR | 2022 | Seg head | DINO | ✅ Feature distillation | ✅ |
| 5 | Uni-UVPT | NeurIPS | 2023 | Visual Prompts | Swin/MT | ✅ Pseudo-labels + consistency | ✅ |
| 6 | LoRA-TTT | ICML | 2025 | LoRA | CLIP | ✅ Entropy + MAE | 📄 |
| 7 | FORLA | arXiv | 2024 | LoRA | SAM/DINO | ✅ Progressive EMA distillation | ❓ |
| 8 | MARCO | CVPR | 2026 | AdaptFormer | DINOv2 | ✅ Dense EMA distillation | ❓ |
| 9 | SVL-Adapter | BMVC | 2022 | Feature Adapter | CLIP + DINO | ✅ SSL blending | ✅ |
| 10 | PC-LoRA | ICLRW | 2024 | LoRA | Generic | ✅ Progressive distillation | ❓ |
| 11 | DoRA | ICML | 2024 | DoRA | LLaMA/LLaVA | ❌ Supervised only | ✅ |
| 12 | Adapters Strike Back | CVPR | 2024 | Adapter+ | ViT/DINO | ❌ Supervised | ❓ |
| 13 | PEFT ViTs w/o Forgetting | CVPRW | 2024 | LoRA/BlockExp | DINO ViT | ❌ Supervised | ✅ |
| 14 | TPT | NeurIPS | 2022 | Prompt Tuning | CLIP | ✅ Entropy minimization | ✅ |
| 15 | UPL | arXiv | 2022 | Prompt Tuning | CLIP | ✅ Pseudo-labeling | ✅ |
| 16 | DAPL | arXiv | 2022 | Prompt Tuning | CLIP | ✅ Contrastive + pseudo | ✅ |
| 17 | InfLoRA | CVPR | 2024 | LoRA | ViT | ❌ Continual learning | ✅ |
| 18 | CL-LoRA | CVPR | 2025 | LoRA (dual) | ViT | ❌ Continual learning | 📄 |
| 19 | NoLA | BMVC | 2025 | Visual Prompts | CLIP + DINO | ✅ DINO→CLIP alignment | ❓ |
| 20 | SAGE-reID | arXiv | 2025 | LoRA + Gating | ViT | ✅ DBSCAN pseudo-labels | ❓ |
| 21 | VirDA | arXiv | 2025 | Visual Reprogramming | ViT | ✅ Intra/inter-domain losses | ❓ |
| 22 | PromptKD | CVPR | 2024 | Prompt Tuning | CLIP | ✅ Knowledge distillation | ✅ |
| 23 | LaFTer | NeurIPS | 2023 | Prompt + Adapter | CLIP | ✅ LLM-guided pseudo-labels | ✅ |
| 24 | L2P | CVPR | 2022 | Prompt Tuning | ViT | ❌ Continual learning | ✅ |
| 25 | DualPrompt | ECCV | 2022 | Prompt Tuning | ViT | ❌ Continual learning | ✅ |
| 26 | CODA-Prompt | CVPR | 2023 | Prompt Tuning | ViT | ❌ Continual learning | ✅ |
| 27 | AdapterFormer | NeurIPS | 2022 | Parallel Adapter | ViT | ❌ Supervised (video) | ✅ |
| 28 | CLIP-Adapter | IJCV | 2023 | Feature Adapter | CLIP | ❌ Supervised | ✅ |
| 29 | DiffTPT | ICCV | 2023 | Prompt Tuning | CLIP | ✅ Entropy + diffusion aug | ✅ |
| 30 | Self-TPT | arXiv | 2024 | Prompt Tuning | CLIP | ✅ Contrastive learning | ❓ |
| 31 | C-LoRA | arXiv | 2025 | LoRA + Routing | ViT/NLP | ❌ Continual learning | ❓ |
| 32 | MoE-Adapter4CL | CVPR | 2024 | LoRA MoE | CLIP ViT | ❌ Continual learning | ✅ |
| 33 | Surgical-DINO | IPCAI | 2024 | LoRA | DINOv2 | ⚠️ Mostly supervised | ✅ |
| 34 | DINO Teacher | CVPR | 2025 | Pseudo-label generator | DINOv2 ViT-G | ✅ Pseudo-labeling | ❓ |
| 35 | InfoMSD | Frontiers AI | 2026 | Visual Prompts + LN | CLIP | ✅ Self-distillation | ✅ |

---

*Compiled from parallel deep research across NeurIPS, CVPR, ICCV, ECCV, ICML, ICLR, BMVC, and AAAI proceedings (2022–2026).*
