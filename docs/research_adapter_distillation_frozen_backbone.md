# Research Report: Self-Distillation & Knowledge Distillation for Adapter Training on Frozen Backbones (Unsupervised)

**Date:** 2026-04-25
**Focus:** A* venue papers (CVPR/ICCV/ECCV/NeurIPS/ICML/ICLR) where:
- A pre-trained backbone remains **frozen**
- Small **adapter modules** (LoRA / bottleneck / prompt) are learned
- Training uses **no ground-truth labels** (self-supervised or unsupervised)
- **Distillation** from a teacher or EMA model is used

---

## 1. Core Methodology Families

### Family A: EMA Teacher-Student Self-Distillation (DINO-style)
A student network with trainable adapters learns to match an EMA (exponential moving average) teacher. The teacher is updated from the student but with stopped gradients.

### Family B: Frozen Teacher → Adapter Student Distillation
A fully frozen teacher (pre-trained foundation model) provides targets. A small student adapter learns to mimic the teacher’s outputs or features.

### Family C: Progressive / Iterative Self-Training with Adapters
Teacher generates pseudo-labels → student adapter trains on pseudo-labels → student becomes new teacher. Backbone remains frozen throughout.

### Family D: Cross-Modal / Multi-Modal Adapter Distillation
Use alignment losses across modalities (RGB, depth, text) with frozen backbone + trainable adapter as the projection head.

---

## 2. Key Papers & Methods

---

### 2.1 DINO — Self-Distillation with No Labels
| | |
|:---|:---|
| **Venue** | ICCV 2021 |
| **Paper** | *Emerging Properties in Self-Supervised Vision Transformers* |
| **Authors** | Caron et al. (Meta AI) |
| **Frozen Backbone?** | No (full end-to-end pre-training), but the **paradigm** is the foundation for all later work |
| **Adapter?** | No — but introduces the EMA teacher-student framework |
| **Labels?** | None |
| **Distillation** | Self-distillation: student predicts EMA teacher output. Teacher updated via `θ_t = m·θ_t + (1-m)·θ_s` |
| **Code** | https://github.com/facebookresearch/dino |
| **Key Insight** | Momentum encoder + multi-crop + centering/sharpening avoids collapse. ViT features naturally encode semantic segmentation maps. |

**Relevance:** The foundational EMA teacher-student mechanism. All subsequent frozen-backbone adapter methods borrow this distillation recipe.

---

### 2.2 STEGO — Unsupervised Semantic Segmentation by Distilling Feature Correspondences
| | |
|:---|:---|
| **Venue** | ICLR 2022 |
| **Paper** | *STEGO: Unsupervised Semantic Segmentation by Distilling Feature Correspondences* |
| **Authors** | Hamilton et al. (MIT / Google / Cornell / Microsoft) |
| **Frozen Backbone?** | **Yes** — DINO ViT backbone is frozen |
| **Adapter?** | Lightweight **segmentation head** (non-linear projection / FFN) acts as an adapter |
| **Labels?** | None |
| **Distillation** | Correspondence distillation: distill feature correlations (self, KNN, random pairs) into compact clusters |
| **Code** | https://github.com/mhamilton723/STEGO |
| **Key Insight** | DINO features already have semantic correspondences. A small head can project them into a space where k-means yields semantic segments. |

**Relevance:** Classic example of frozen backbone + lightweight head trained with unsupervised distillation loss. Trains in <2 hours on a single V100.

---

### 2.3 LaFTer — Label-Free Tuning of Zero-Shot Classifiers
| | |
|:---|:---|
| **Venue** | NeurIPS 2023 |
| **Paper** | *LaFTer: Label-Free Tuning of Zero-shot Classifier using Language and LLMs* |
| **Authors** | Mirza et al. |
| **Frozen Backbone?** | **Yes** — CLIP vision encoder frozen |
| **Adapter?** | **Visual Prompt Tuning (VPT)** + LayerNorm scale/shift parameters (<0.4% of total params) |
| **Labels?** | None |
| **Distillation** | Self-training / pseudo-labeling: text-only pre-trained classifier generates pseudo-labels for unlabeled images; student refines via consistency regularization (FixMatch-style) |
| **Code** | (available in paper supplementary) |
| **Key Insight** | A text-only classifier trained on LLM-generated class descriptions generates pseudo-labels. Visual prompts adapt the frozen CLIP encoder without labels. Outperforms CLIP-PR and UPL. |

**Relevance:** Purely unsupervised adapter tuning (prompts + LN) via self-generated pseudo-labels. No ground-truth labels needed.

---

### 2.4 Omnivorous DINOv2 — Cross-Modal Adapter Distillation
| | |
|:---|:---|
| **Venue** | CVPR 2026 (arXiv 2025) |
| **Paper** | *A Mixed Diet Makes DINO An Omnivorous Vision Encoder* |
| **Authors** | (Probe3D / DINOv2 follow-up team) |
| **Frozen Backbone?** | **Yes** — DINOv2 ViT backbone frozen (first L blocks); only adapter trainable |
| **Adapter?** | **Zero-initialized adapter network** (4 ViT blocks) on top of frozen backbone; OR fine-tune final 4 blocks |
| **Labels?** | None — multimodal contrastive / distillation loss |
| **Distillation** | Teacher = frozen DINOv2 with frozen head. Student = frozen backbone + trainable adapter g. Loss = distill teacher outputs + cross-modal alignment (RGB ↔ Depth ↔ Seg) |
| **Code** | (to be released; check Probe3D repo) |
| **Key Insight** | Teacher head remains frozen as a stable anchor. Distilling from teacher while aligning across modalities prevents catastrophic forgetting. Adapter-on-top matches fine-tuning-last-blocks performance. |

**Relevance:** Directly demonstrates training an **adapter on top** of a frozen DINOv2 backbone via teacher-student distillation. The adapter learns modality-invariant embeddings.

---

### 2.5 KD-LoRA — Knowledge Distillation + LoRA
| | |
|:---|:---|
| **Venue** | arXiv 2024 (highly cited) |
| **Paper** | *KD-LoRA: A Hybrid Approach to Efficient Fine-Tuning with LoRA and Knowledge Distillation* |
| **Authors** | Azimi et al. |
| **Frozen Backbone?** | Student backbone frozen; LoRA modules injected into student |
| **Adapter?** | **LoRA** (Low-Rank Adaptation) modules in student |
| **Labels?** | Supervised task data used for teacher; student learns via KD |
| **Distillation** | Full fine-tuned teacher → student with LoRA. Feature/logit distillation. |
| **Code** | https://github.com/rambodazimi/KD-LoRA |
| **Key Insight** | Combines parameter efficiency (LoRA) with KD. Student is smaller + LoRA-injected. 98% of LoRA performance, 40% more compact, 30% less GPU memory. |

**Relevance:** Shows how to distill from a teacher into a **LoRA-augmented student** with frozen backbone. Primarily NLP but applicable to vision.

---

### 2.6 PC-LoRA — Progressive Compression via LoRA + Distillation
| | |
|:---|:---|
| **Venue** | ICLR 2024 Workshop / arXiv 2024 |
| **Paper** | *PC-LoRA: Low-Rank Adaptation for Progressive Model Compression with Knowledge Distillation* |
| **Authors** | Hwang et al. (C-LoRA LAB / KAIST) |
| **Frozen Backbone?** | Pre-trained weights frozen initially, then **gradually decayed to zero** |
| **Adapter?** | **LoRA** only — eventually only adapters remain |
| **Labels?** | Standard supervised task loss + distillation loss |
| **Distillation** | Feature-based KD between original teacher (full model) and student (LoRA-only). Progressive decay of frozen weights via scheduled λ. |
| **Code** | (OpenReview) |
| **Key Insight** | Achieves **94.36% parameter compression** and **89.1% FLOPs reduction** for ViT-B by making LoRA adapters self-sufficient. The frozen backbone is effectively distilled into the adapters. |

**Relevance:** Proof that a frozen backbone’s knowledge can be progressively distilled into tiny LoRA adapters, making them standalone.

---

### 2.7 FORLA — Federated Object-Centric Representation Learning with Self-Distillation
| | |
|:---|:---|
| **Venue** | arXiv 2024 / ICLR 2025 (federated learning) |
| **Paper** | *FORLA: Federated Object-Centric Representation Learning with Slot Attention* |
| **Authors** | Liao et al. (UPenn) |
| **Frozen Backbone?** | Foundation model (SAM/DINO/MAE) features frozen; adapter + slot attention are trainable |
| **Adapter?** | MLP / MOE / AFM (Attention Feature Modulation) adapter |
| **Labels?** | None — unsupervised slot attention + reconstruction |
| **Distillation** | **Progressive two-stage self-distillation**: (1) EMA teacher adapter updated from student adapter; (2) later, bidirectional knowledge transfer via FedAvg. Teacher decoder gradients blocked early. |
| **Code** | (check paper for repo) |
| **Key Insight** | Self-distillation with EMA teacher adapter dramatically improves performance (e.g., Abdominal CorLoc 29.22 → 79.40). MOE adapters are most robust. |

**Relevance:** Strong example of **EMA-based self-distillation specifically for adapters** on frozen foundation features. Progressive training prevents collapse.

---

### 2.8 DINO Teacher — Domain Adaptive Object Detection with Frozen DINOv2
| | |
|:---|:---|
| **Venue** | CVPR 2025 |
| **Paper** | *Large Self-Supervised Models Bridge the Gap in Domain Adaptive Object Detection* |
| **Authors** | (CVPR 2025) |
| **Frozen Backbone?** | **Yes** — DINOv2 ViT-G frozen as encoder for labeller |
| **Adapter?** | Detector head (Faster R-CNN) trained on frozen DINOv2 features |
| **Labels?** | Source labels only; target domain unlabeled |
| **Distillation** | Frozen labeller generates pseudo-labels for target domain. Student trained with source GT + target pseudo-labels. Patch feature alignment to frozen DINOv2. |
| **Code** | (check CVPR 2025 proceedings) |
| **Key Insight** | DINOv2 generalizes across domain gaps better than EMA student teachers. Frozen backbone + simple detection head outperforms Mean Teacher by ~5% AP. |

**Relevance:** Demonstrates that a **frozen self-supervised backbone** (DINOv2) is a better pseudo-label generator than an EMA student for domain adaptation.

---

### 2.9 MARCO — Dense Self-Distillation for Semantic Correspondence
| | |
|:---|:---|
| **Venue** | CVPR 2026 (likely, based on arXiv) |
| **Paper** | *Navigating the Unseen Space of Semantic Correspondence* (MARCO) |
| **Authors** | (arXiv 2026) |
| **Frozen Backbone?** | **Yes** — DINOv2 frozen |
| **Adapter?** | **AdaptFormer** adapter layers (bottleneck adapters in ViT) + feature upsampling |
| **Labels?** | Sparse keypoint supervision only; dense regions learned via self-distillation |
| **Distillation** | **Dense self-distillation via EMA teacher**: teacher generates pseudo-correspondences from frozen DINOv2 features. EMA teacher updated with momentum m∈[0.996, 0.999]. |
| **Code** | (to be released) |
| **Key Insight** | Adding dense self-distillation improves generalization to unseen keypoints (SPair-U PCK@0.10: 42.0 → 67.5). AdaptFormer outperforms LoRA and standard Adapter. |

**Relevance:** Shows **AdaptFormer adapters** on frozen DINOv2 trained with EMA self-distillation for dense prediction tasks.

---

### 2.10 InfoMSD — Self-Distillation for Parameter-Efficient Fine-Tuning (Artwork)
| | |
|:---|:---|
| **Venue** | Frontiers in Artificial Intelligence, 2026 |
| **Paper** | *InfoMSD: Information-Maximization Self-Distillation Framework for Parameter-Efficient Fine-Tuning* |
| **Authors** | (Frontiers 2026) |
| **Frozen Backbone?** | **Yes** — CLIP ViT-B/32 frozen |
| **Adapter?** | **Visual prompts** (learnable tokens prepended to patch embeddings) + LayerNorm parameters |
| **Labels?** | None |
| **Distillation** | Two-stage: (1) Zero-shot teacher generates pseudo-labels; (2) Student with visual prompts learns via cross-entropy + entropy regularization. |
| **Code** | Dassl.pytorch framework |
| **Key Insight** | Less than 1% trainable parameters. Self-distillation from zero-shot teacher + strong augmentations achieves label-free adaptation. |

**Relevance:** Directly applicable recipe: frozen CLIP/DINO + visual prompts + self-distillation from zero-shot teacher.

---

### 2.11 Lite Prompted Self-Training (Adapter-Based Self-Training)
| | |
|:---|:---|
| **Venue** | NAACL 2022 Findings |
| **Paper** | *Lite Prompted Self-training Makes Parameter-efficient Few-shot Learners* |
| **Authors** | (Purdue / NAACL) |
| **Frozen Backbone?** | **Yes** — PLM (e.g., RoBERTa) encoder frozen for both teacher and student |
| **Adapter?** | **Adapter layers** in both teacher and student |
| **Labels?** | Few-shot labeled data for teacher init; then self-training on unlabeled data |
| **Distillation** | Iterative: (1) Train teacher adapter on few-shot labels; (2) Generate pseudo-labels; (3) Meta-reweight noisy pseudo-labels; (4) KD warmup for student adapter; (5) Student becomes teacher. |
| **Code** | (paper repo) |
| **Key Insight** | KD warmup stabilizes student adapter training. Student adapter re-initialized each iteration to prevent label leakage. Shared encoder always frozen. |

**Relevance:** Classic iterative self-training with adapters on frozen backbones, using KD as a warmup mechanism.

---

### 2.12 Freeze the Backbones — Medical VLP with Adaptor
| | |
|:---|:---|
| **Venue** | arXiv 2024 (Medical Imaging) |
| **Paper** | *Freeze the backbones: A Parameter-Efficient Contrastive Approach to Robust Medical Vision-Language Pre-training* |
| **Authors** | (Medical imaging group) |
| **Frozen Backbone?** | **Yes** — DINOv2 and BERT frozen |
| **Adapter?** | **Cross-attention Adaptor module** (lightweight, backbone-agnostic) |
| **Labels?** | None — contrastive learning on image-text pairs |
| **Distillation** | Not distillation per se, but cross-modal contrastive alignment via adapter. Backbone embeddings pre-computed for efficiency. |
| **Code** | (check paper) |
| **Key Insight** | >90% reduction in trainable parameters vs. end-to-end. Training takes ~15 minutes on 2× Tesla T4. Performance rivals full end-to-end medical VLP. |

**Relevance:** Shows that even simple contrastive alignment through adapters on frozen backbones is highly effective and extremely efficient.

---

### 2.13 UINO-FSS — Hierarchical Distillation + Bottleneck Adapter
| | |
|:---|:---|
| **Venue** | arXiv 2025 |
| **Paper** | *UINO-FSS: Unifying Representation Learning and Few-shot Segmentation via Hierarchical Distillation* |
| **Authors** | (arXiv 2025) |
| **Frozen Backbone?** | **Yes** — DINOv2 encoder frozen |
| **Adapter?** | **Bottleneck Adapter (BA)** + Meta-Visual Prompt Generator |
| **Labels?** | Few-shot labels for decoder; adapter trained via distillation |
| **Distillation** | Two-stage: (1) MSE distillation from SAM encoder to DINOv2 adapter; (2) Mask decoder trained with frozen encoder + adapter. |
| **Code** | (to be released) |
| **Key Insight** | Hierarchical distillation into bottleneck adapter bridges the gap between DINOv2 and SAM features. Removing distillation drops mIoU by 6.8%. |

**Relevance:** Feature-level distillation from a strong teacher (SAM) into a bottleneck adapter on frozen DINOv2.

---

### 2.14 MultiFCL — Federated Continual Learning with Adapter Self-Distillation
| | |
|:---|:---|
| **Venue** | NeurIPS 2025 |
| **Paper** | *Federated Continual Learning via Orchestrating Multi-Scale Expertise* |
| **Authors** | (NeurIPS 2025) |
| **Frozen Backbone?** | Adapters frozen after training; backbone fine-tuned then frozen |
| **Adapter?** | Lightweight task adapters |
| **Labels?** | Task-specific labels |
| **Distillation** | **Multi-expert dynamic self-distillation** with multi-scale feature learning. Intra-client and inter-client expert communication. |
| **Code** | (NeurIPS 2025 proceedings) |
| **Key Insight** | Adapters are learned, then frozen for stability. Self-distillation between experts enables cross-task knowledge fusion without rehearsal. |

---

### 2.15 DINOv3 / SegDINO / MedDINOv3
| | |
|:---|:---|
| **Venue** | Various 2025-2026 |
| **Papers** | DINOv3 family (arXiv 2025), SegDINO, MedDINOv3 |
| **Frozen Backbone?** | **Yes** — DINOv3 backbone almost always frozen in downstream usage |
| **Adapter?** | Decoder / adapter / LoRA (e.g., 650k params for MIDOG classification) |
| **Labels?** | Varies; self-supervised pre-training uses none |
| **Distillation** | DINOv3 itself uses self-distillation (EMA teacher + Gram anchoring loss). Downstream: frozen backbone + adapter/decoders. |
| **Code** | https://github.com/facebookresearch/dinov3 (expected) |
| **Key Insight** | DINOv3 continues the self-distillation paradigm at scale (1.7B images). The frozen-backbone + adapter paradigm is now standard practice. |

---

## 3. Methods Summary Table

| Paper | Venue | Backbone | Adapter Type | Labels | Distillation Type | Code |
|:---|:---|:---|:---|:---|:---|:---|
| **DINO** | ICCV 2021 | ViT/ResNet | None (full model) | None | EMA self-distillation | [github](https://github.com/facebookresearch/dino) |
| **STEGO** | ICLR 2022 | DINO ViT (frozen) | Segmentation head | None | Feature correspondence distillation | [github](https://github.com/mhamilton723/STEGO) |
| **LaFTer** | NeurIPS 2023 | CLIP ViT (frozen) | Visual prompts + LN | None | Self-training / pseudo-labeling | (paper) |
| **Omnivorous DINOv2** | CVPR 2026 | DINOv2 ViT (frozen) | ViT adapter blocks (4) | None | Teacher-student + cross-modal | (Probe3D) |
| **KD-LoRA** | arXiv 2024 | BERT/RoBERTa/DeBERTa | LoRA | Supervised | Logit/feature KD | [github](https://github.com/rambodazimi/KD-LoRA) |
| **PC-LoRA** | ICLR 2024W | ViT-B / BERT | LoRA | Supervised | Feature KD + progressive decay | (OpenReview) |
| **FORLA** | arXiv 2024 | SAM/DINO/MAE (frozen) | MLP/MOE/AFM adapter | None | EMA progressive self-distillation | (UPenn) |
| **DINO Teacher** | CVPR 2025 | DINOv2 ViT-G (frozen) | Detector head | Source only | Pseudo-label generation | (CVPR 2025) |
| **MARCO** | CVPR 2026 | DINOv2 (frozen) | AdaptFormer | Sparse only | EMA dense self-distillation | (to be released) |
| **InfoMSD** | Frontiers 2026 | CLIP ViT (frozen) | Visual prompts | None | Zero-shot teacher pseudo-labels | Dassl |
| **Lite Prompted ST** | NAACL 2022 | RoBERTa (frozen) | Adapters | Few-shot + unlabeled | Iterative self-training + KD warmup | (paper) |
| **Freeze the Backbones** | arXiv 2024 | DINOv2 + BERT (frozen) | Cross-attention adaptor | None | Cross-modal contrastive | (paper) |
| **UINO-FSS** | arXiv 2025 | DINOv2 (frozen) | Bottleneck Adapter | Few-shot | Feature KD from SAM | (to be released) |
| **MultiFCL** | NeurIPS 2025 | PTM (frozen adapters) | Task adapters | Task labels | Multi-expert self-distillation | (NeurIPS) |

---

## 4. Common Implementation Recipes

### Recipe 1: EMA Teacher + Student Adapter (DINO-style)
```python
# Student: frozen backbone + trainable adapter
student = FrozenBackboneWithAdapter(backbone, adapter)
# Teacher: EMA of student (including adapter)
teacher = create_ema_model(student, momentum=0.996)

for x in loader:
    # Multi-crop: global views → teacher, all views → student
    student_out = student(augment(x))
    with torch.no_grad():
        teacher_out = teacher(augment_global(x))
    
    loss = cross_entropy(student_out, softmax(teacher_out / temp_t))
    # Only student adapter gradients; teacher updated via EMA
```

### Recipe 2: Frozen Teacher → Trainable Adapter Student
```python
teacher = load_pretrained_frozen_model()  # e.g., DINOv2
student = AdapterOnTopOfFrozenBackbone(backbone, adapter_layers)

for x in loader:
    with torch.no_grad():
        teacher_feat = teacher(x)
    student_feat = student(x)
    
    loss = mse_loss(student_feat, teacher_feat)  # feature distillation
    # OR: logit distillation, or pseudo-label cross-entropy
```

### Recipe 3: Progressive Self-Training with Adapters (Iterative)
```python
for iteration in range(M):
    # Step 1: Teacher generates pseudo-labels on unlabeled data
    pseudo_labels = teacher_model(unlabeled_data)
    
    # Step 2: Student adapter trains on pseudo-labels (+ KD warmup)
    student_adapter = train_adapter(backbone, pseudo_labels, warmup_kd=True)
    
    # Step 3: Student becomes new teacher
    teacher_model = copy(student_adapter)
```

### Recipe 4: Cross-Modal Adapter Alignment
```python
# Frozen backbones for RGB, depth, etc.
backbone_rgb = frozen_dinov2()
backbone_depth = frozen_dinov2()  # or same shared backbone

# Trainable adapter projects to shared space
adapter = ModalityAgnosticAdapter()

for rgb, depth, seg in multimodal_loader:
    z_rgb = adapter(backbone_rgb(rgb))
    z_depth = adapter(backbone_depth(depth))
    z_seg = adapter(backbone_rgb(seg))  # or seg backbone
    
    loss = info_nce(z_rgb, z_depth) + info_nce(z_rgb, z_seg)
    # Distill from frozen teacher head to preserve semantics
```

---

## 5. Key Findings for Your Project

1. **EMA teachers are the gold standard** for stabilizing adapter training without labels. Momentum coefficients typically in `[0.996, 0.999]`.

2. **Frozen backbone + adapter is extremely efficient.** Training time drops from days to hours (e.g., STEGO <2h on V100; Freeze-the-Backbones ~15 min on 2× T4).

3. **Adapter placement matters.** Late-layer adapters (e.g., final 4 ViT blocks) capture high-level semantics; early adapters capture structure. Some methods train adapters **on top** of the entire frozen backbone with comparable results.

4. **Progressive distillation prevents collapse.** FORLA and PC-LoRA both use staged training: early EMA-only updates, later explicit distillation or weight decay.

5. **Pseudo-labeling from frozen teachers outperforms EMA students** when domain gaps exist (DINO Teacher paper). A strong frozen foundation model is often the best teacher.

6. **LoRA is not always best.** For dense prediction, AdaptFormer and bottleneck adapters sometimes outperform LoRA (see MARCO ablations: AdaptFormer 87.2 PCK vs LoRA 85.2).

7. **Self-distillation can be done entirely with adapters.** PC-LoRA proves that the frozen backbone can be progressively removed, leaving only adapters as the compressed model.

---

## 6. Recommended Papers to Read First

| Priority | Paper | Why |
|:---|:---|:---|
| **1** | DINO (ICCV 2021) | Foundation of EMA self-distillation |
| **2** | STEGO (ICLR 2022) | Frozen backbone + lightweight head + unsupervised distillation |
| **3** | Omnivorous DINOv2 (CVPR 2026) | Frozen DINOv2 + adapter-on-top + teacher-student distillation |
| **4** | FORLA (arXiv 2024) | EMA progressive self-distillation for adapters explicitly |
| **5** | PC-LoRA (ICLR 2024W) | Distilling backbone into LoRA adapters progressively |
| **6** | LaFTer (NeurIPS 2023) | Unsupervised adapter tuning via pseudo-labeling |
| **7** | MARCO (CVPR 2026) | AdaptFormer + dense EMA self-distillation on frozen DINOv2 |

---

*Report compiled for MBPS project — adapter distillation research track.*
