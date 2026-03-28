Research Results: Training a New Backbone Without CUPS Recipes                                                                                       
                                                                                                                                                    
Yes, this is very feasible. The agent analyzed 5 approaches and the existing codebase. Here's the key finding:                                            
                                                                                                                                                    
The Bottleneck                                                                                                                                            
                                                                                                                                                        
Semantics are solved (PQ_stuff=35.13). The entire gap is in instances (PQ_things=19.92), especially person (PQ=4.2, only 170/3206 matched) — depth-guided 
splitting can't separate co-planar objects.                                                                                                               
                                                                                                                                                        
Top Approach: DINOv2-UNet Panoptic (Approach 1)                                                                                                           
                                                                                                                                                        
Extend your existing UNet P2-B with lightweight center heatmap + offset heads (Panoptic-DeepLab style):                                                   
                                                                                                                                                        
┌───────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────────┐                                  
│    Aspect     │                                                Detail                                                │                                  
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Base          │ Pre-trained UNet P2-B checkpoint (PQ=28.00)                                                          │                                  
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ New heads     │ Center heatmap (1ch) + offset (2ch), ~50 lines of code                                               │                                  
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Training      │ Phase 1: freeze UNet, train instance heads only (10 ep). Phase 2: fine-tune all (10 ep, 0.1x LR)     │                                  
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Targets       │ Generated from existing pseudo_instance_spidepth/ via generate_instance_targets.py (already exists!) │                                  
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Inference     │ panoptic_inference_center_offset() already implemented in panoptic_deeplab.py                        │
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Code reuse    │ ~80% — loss, inference, target gen, dataset all exist                                                │
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Training time │ ~4-6 hours on 1x GTX 1080 Ti                                                                         │
├───────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────────┤                                  
│ Expected PQ   │ 29-30 (PQ_things ~22-24)                                                                             │
└───────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────────┘                                  
                
Why this differs from the failed RepViT instance heads:                                                                                                   
1. DINOv2 features (768-dim) vs weak RepViT features
2. Windowed attention at 128x256 vs simple FPN at 32x64                                                                                                   
3. Curriculum training (freeze semantic → add instances) vs joint from scratch
                                                                                                                                                        
Follow-up: Iterative Self-Training (Approach 2)                                                                                                           
                                                                                                                                                        
After Approach 1, use the model's improved instance predictions as new pseudo-labels and retrain 2-3 rounds (CutLER-style). Each round: +1-2 PQ_things →  
could reach PQ 29.5-31.
                                                                                                                                                        
Quick Wins (Zero Training)                                                                                                                                

Approach 5: DINOv2 feature clustering within connected components. Two co-planar persons have different features → k-means can split them. Zero training  
cost, +0.5-1.5 PQ_things.
                                                                                                                                                        
Key Risks & Mitigations

- Noisy targets: Use DINOv2 cosine similarity as auxiliary loss for appearance-based separation                                                           
- Semantic degradation: Freeze semantic path in Phase 1, use cascade gradient scaling
- Small objects: Adaptive sigma for heatmaps, upsample offsets for full-res grouping                                                                      
                                                                                                                                                        
What NOT to do                                                                                                                                            
                                                                                                                                                        
- Mask2Former-Lite on pseudo-labels (too sensitive to label noise)                                                                                        
- EMA self-training (already proven to hurt with your labels)
                                                                                                                                                        
---                                                                                                                                                       
Bottom line: You already have ~80% of the code. The main new work is ~100 lines to extend the UNet with instance heads and integrate losses. Want me to   
implement Approach 1?