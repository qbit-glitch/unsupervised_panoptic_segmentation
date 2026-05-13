---             
Abstract
        
▎ Unsupervised panoptic segmentation must recover both pixel-level semantics and individual object identities without human masks, and the strongest 
▎ scene-centric recipe to date, CUPS, supplies the missing structure from stereo depth and scene flow at pseudo-label time — a commitment that excludes   
▎ the monocular footage in which most images actually arrive. We ask whether a single frame, read carefully, carries the same structure, and find that it 
▎ does, provided no individual cue is trusted alone. We reframe pseudo-label construction as a problem of cross-modal agreement: frozen semantic codes    
▎ propose categories, monocular depth proposes physical separations, and a guarded filter accepts a label only when both cues concur. Two compact 
▎ mechanisms instantiate this principle inside an otherwise standard pipeline. DCFA is a 40K-parameter zero-initialized residual that bends a frozen
▎ 90-dimensional CAUSE-TR code toward plausible geometry through a 16-dimensional sinusoidal depth embedding; SIMCF-ABC enforces semantic uniformity
▎ inside each instance proposal, merges adjacent depth fragments only when class identity and DINOv3 cosine similarity agree (τ=0.85), and rejects pixels
▎ whose depth violates per-class statistics. On Cityscapes, DCFA lifts semantic clustering mIoU from 52.69 to 55.29, and the full pipeline raises
▎ pseudo-label panoptic quality from 24.54 to 25.85, with the gain concentrated on the things split (PQ_th 12.31 → 14.70). Trained with the published CUPS
▎  bootstrapping and self-training recipe on a frozen DINOv3 ViT-B/16 Cascade Mask R-CNN, the resulting detector reaches 35.83 PQ on the Cityscapes
▎ validation set, against 27.80 PQ for the published CUPS configuration (ResNet-50, stereo+motion) and 32.76 PQ for the same recipe trained on monocular
▎ pseudo-labels without our cross-modal filter; the +3.07 PQ contribution-isolated gap names what monocular agreement adds under identical training.

---
Sentence-level rhetorical breakdown
                                                                                                                                                        
┌─────┬────────────────────────────┬───────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────┐
│  #  │            Move            │        Stress position        │                              Why it earns its keep                               │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ 1   │ Context + specific Gap     │ "monocular footage in which   │ Names the exact prerequisite of CUPS (stereo depth + scene flow at pseudo-label  │   
│     │                            │ most images actually arrive"  │ time) and ends on the deployment reality that breaks it. No generic opener.      │
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
│ 2   │ Insight (delayed           │ "no individual cue is trusted │ Holds curiosity open one sentence — the reader does not yet know how.            │
│     │ mechanism)                 │  alone"                       │                                                                                  │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ 3   │ Principle named            │ "only when both cues concur"  │ The agreement principle is stated before any component is named.                 │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤   
│ 4   │ Bridge                     │ "an otherwise standard        │ Sets the scope honestly: we add two mechanisms, not a new architecture.          │
│     │                            │ pipeline"                     │                                                                                  │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│     │ Mechanism, content-driven  │ "depth violates per-class     │ Each component named with its concrete operating detail (40K params, 16-D        │   
│ 5   │ triple                     │ statistics"                   │ sinusoidal, τ=0.85). The SIMCF triple mirrors stages A/B/C — content-driven, not │
│     │                            │                               │  rule-of-three filler.                                                           │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ 6   │ Evidence (Stage-1,         │ "PQ_th 12.31 → 14.70"         │ Stress position holds the per-split decomposition that proves the gain is on     │
│     │ decomposed)                │                               │ things, the bottleneck the gap named.                                            │   
├─────┼────────────────────────────┼───────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────┤
│ 7   │ Punch +                    │ "monocular agreement adds     │ Final clause names what is contributed after the backbone is held fixed (35.83   │   
│     │ contribution-isolation     │ under identical training"     │ vs 32.76, both on DINOv3 ViT-B/16). Pre-empts the backbone-confound reviewer.    │   
└─────┴────────────────────────────┴───────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────┘
                                                                                                                                                        
Tension Arc: Gap → Insight → Mechanism → Evidence → Punch ✓                                                                                               
Em-dashes: 1 (within the cvpr-narrative budget).
Anti-AI audit: no "crucial role / paves the way / stands as a testament"; no vague attributions; no copula-avoidance ("serves as", "represents"); no      
intensifiers ("very", "really", "significantly"); zero rule-of-three rhetoric (the SIMCF triple is content-driven).                                       
Word count: ~265 words (NeurIPS-typical range 200-300).                                                                                                   
                                                                                                                                                        
---             
Verification ledger — every number now grounded in a source file                                                                                          
                                                                                                                                                        
┌────────────────────────────────────────┬───────────────────────────────────────────────────────────────────────────────┬───────────────────────────┐ 
│                 Claim                  │                                Source on disk                                 │          Status           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ CUPS published 27.80 PQ on Cityscapes  │ refs/cups/README.md Table 1; Research/.../Papers/CUPS-2025.md:59              │ ✅ verified               │ 
│ val (ResNet-50, stereo+motion)         │                                                                               │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ DCFA: 40K params, 90-D CAUSE-TR code,  │ reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md:45-53; checkpoint       │ ✅ verified — fixes the   │ 
│ 16-D sinusoidal, h=384                 │ results/depth_adapter/V3_dd16_h384_l2/best.pt                                 │ existing draft's "h=128"  │    
│                                        │                                                                               │ inconsistency in §3.3     │ 
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ SIMCF-B cosine threshold τ=0.85        │ reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md SIMCF-B subsection;     │ ✅ verified               │ 
│                                        │ existing draft §3.5                                                           │                           │ 
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ Semantic mIoU 52.69 → 55.29 on         │                                                                               │                           │ 
│ Cityscapes (K=80 adapter-eval          │ reports/depth_semantic_ablation_complete.md:191 (Table line 155)              │ ✅ verified               │    
│ protocol)                              │                                                                               │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
│ Pseudo-label PQ 24.54 → 25.85          │ reports/dcfa_depthpro_simcf_abc_pseudolabel_report.md:241,245 (rows A0 and    │ ✅ verified               │    
│                                        │ A5)                                                                           │                           │ 
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ PQ_th 12.31 → 14.70 (gain on things    │ same report, rows A0 and A5                                                   │ ✅ verified               │ 
│ split)                                 │                                                                               │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
│ 35.83 PQ on Cityscapes val (saved-best │ results/stage3_dcfa_simcf_abc_step3000_eval.json (PQ: 0.35832); checkpoint    │ ✅ verified, single seed  │    
│  Stage-3, step 3000)                   │ checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt                    │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤ 
│ 27-class CAUSE + Hungarian protocol    │ refs/cups/cups/metrics/panoptic_quality.py; project memory rule "CUPS eval    │ ✅ verified               │    
│                                        │ protocol"                                                                     │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ 32.76 PQ same-backbone vanilla         │ reports/dinov3_cups_results.md:45; BMVC LaTeX abstract                        │                           │ 
│ baseline (DINOv3 ViT-B/16, k=80 + DA3, │ paper_bmvc2026/main.tex:39                                                    │ ✅ verified               │    
│  no DCFA/SIMCF)                        │                                                                               │                           │    
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤
│ Backbone-isolated effect: ResNet-50    │ reports/neurips_narrative_report.md:271-273;                                  │                           │    
│ 24.68 → DINOv3 ViT-B/16 27.87          │ reports/dinov3_cups_results.md:27                                             │ ✅ verified               │    
│ (Stage-2)                              │                                                                               │                           │
├────────────────────────────────────────┼───────────────────────────────────────────────────────────────────────────────┼───────────────────────────┤    
│ +3.07 PQ contribution-isolated (35.83  │                                                                               │                           │
│ − 32.76, same backbone, same recipe,   │ derived from the two verified rows above                                      │ ✅ verified arithmetic    │    
│ only pseudo-labels differ)             │                                                                               │                           │
└────────────────────────────────────────┴───────────────────────────────────────────────────────────────────────────────┴───────────────────────────┘    

---
Numbers I deliberately kept out of the abstract
                                                                                                                                                        
┌───────────────────────────┬─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┐
│          Number           │                                                      Why excluded                                                       │   
├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Peak PQ = 39.12 at step   │ Checkpoint not saved (save_top_k=1 with DDP sync bug); from a different training branch (warm-start from step 2200, not │
│ ~2600 (Seesaw ablation    │  the main 35.83 chain). Unrecoverable.                                                                                  │
│ branch)                   │                                                                                                                         │
├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Mapillary zero-shot PQ =  │ Strong number, but cross-dataset transfer belongs in §5, not the abstract. The abstract should land on Cityscapes, the  │
│ 39.19                     │ controlled setting.                                                                                                     │
├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ DepthPro PQ_th = 23.35    │ The 35.83 saved Stage-3 checkpoint was trained on DA3 instance labels per its config name ..._dcfa_da3_simcf_abc_....   │
│ (best Stage-1             │ Citing DepthPro as the headline depth model would mismatch the actual trained checkpoint. The Stage-1 best-pseudo-label │
│ pseudo-label PQ)          │  result (DepthPro) is a separate claim that belongs in §4 / Table 3, not the abstract.                                  │
├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ Single-seed disclaimer    │ Belongs in Limitations §6 (todo_neurips_e2_multiseed_stage3 is on your blocking-todo list). Not abstract material.      │
├───────────────────────────┼─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┤
│ E1 same-backbone          │                                                                                                                         │
│ CUPS-stereo control still │ Belongs in Limitations §6.                                                                                              │
│  blocked                  │                                                                                                                         │
└───────────────────────────┴─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────┘

---
Three open decisions only you can make

1. Should the headline name a specific depth model? Right now the abstract says "monocular depth" generically. If you want to anchor on DA3 (matches the
trained checkpoint), I'd insert "Depth Anything v3" once. If you want to anchor on DepthPro (matches the best Stage-1 result), I'd add a clause clarifying
that the trained Stage-3 detector uses a different depth model than the best Stage-1. The current generic phrasing avoids the tension.
2. The +3.07 PQ contribution-isolated framing — I added this to pre-empt the backbone-confound attack. If you want a tighter, less-defensive abstract,
drop the ", and 32.76 PQ for the same recipe trained on monocular pseudo-labels without our cross-modal filter; the +3.07 PQ contribution-isolated gap
names what monocular agreement adds under identical training" tail and end on "stereo+motion)". The abstract becomes ~30 words shorter.
3. DCFA h=384 correction. The existing §3.3 of paper/mbps_neurips2026_full_draft.md says h=128. The verified canonical value is h=384. Want me to also
patch §3.3 to match the abstract, or leave that for when we draft the Method section?