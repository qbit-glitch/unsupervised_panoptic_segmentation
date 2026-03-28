## Question #1: 

To cross PQ=25, you need an instance head. Semantic-only connected components fundamentally cap PQ_things. Two adjacent cars of the same class will always merge into one segment. No amount of augmentation or batch size fixes this — it's a structural limitation.                                         
                                                                                                                                                    
let's discuss: 
1. what is the significance of instance head ?
2. Does it refine our instance labels just like the unet adoption for the refinement of the semantic labels ?                                         
3. What can be the possible architecture for refining our instances ?                                                                                     
                                                                                                                                                        
#### 1. What does an instance head do?                                         
                                                                                                                                                        
A semantic head answers: "What class is this pixel?" (road, car, person...).                                                                 
An instance head answers: "Which specific object is this pixel?" (car #1 vs car #2 vs car #3)                                                             
                                                          
Without an instance head, if two cars are touching, the semantic map says "car, car, car" for all pixels — and connected components merges them into one segment. PQ penalizes this as 1 TP + 1 FN instead of 2 TPs.                 
                                                                     
There are two approaches:

a. Detection-based (what CUPS uses): Predict bounding boxes first, then predict a binary mask inside each box. This is Cascade Mask R-CNN — heavy, ~36M
params for the heads alone.

b. Embedding-based (lightweight alternative): Predict a 16-dim embedding vector per pixel. Pixels belonging to the same object have similar embeddings,
different objects have distant embeddings. Then cluster at inference (mean-shift or simple distance thresholding). This is ~0.1-0.3M params extra.

2. Is it like UNet refinement for instances?

No, fundamentally different.

UNet semantic refinement: Takes noisy 19-class pseudo-labels → outputs cleaner 19-class predictions. It's a denoising task — the classes already exist,
you're just fixing boundaries.

Instance head: Creates something that doesn't exist in the input. Your pseudo-labels have semantic classes but the instance separation is weak
(depth-guided splitting at PQ_things=19.41). The instance head must learn to separate individual objects — this is a generative task, not refinement.

That said, you could frame instance refinement similarly to semantic refinement:
- Input: coarse instance pseudo-labels (from depth-guided splitting)
- Output: refined instance boundaries

But the problem is harder because instance IDs are arbitrary (car #1 and car #2 are interchangeable), so you can't use simple CE loss. You need either:
- Discriminative embedding loss: pull same-instance pixels together, push different-instance pixels apart
- Boundary regression: predict instance boundary maps, then watershed
- Offset regression: predict 2D offset vectors pointing to each object's center

3. Possible architectures for instance refinement

Three options, increasing in complexity:

Option A: Embedding Head (simplest, recommended)
RepViT backbone → SimpleFPN → [semantic head (19-class)]
                            → [embedding head (16-dim)]

Training: discriminative loss on pseudo-instance labels
Inference: mean-shift clustering on embeddings → instance IDs
- Extra params: ~0.1M (one depthwise-separable conv + 1x1 conv)
- Advantage: No detection pipeline needed, works with any backbone
- Used by: Panoptic-DeepLab [Cheng et al., CVPR 2020], CUPS baseline

Option B: Center + Offset Head (Panoptic-DeepLab style)
RepViT backbone → SimpleFPN → [semantic head (19-class)]
                            → [center heatmap head (1-ch)]
                            → [offset head (2-ch)]

Training: MSE loss on GT center heatmaps + L1 on offset vectors
Inference: find center peaks → group pixels by nearest center
- Extra params: ~0.15M
- Advantage: Deterministic grouping (no clustering needed), faster inference
- Advantage: Center heatmap gives "objectness" score for free
- Used by: Panoptic-DeepLab, Real-Time Panoptic Segmentation [Hou et al.]

Option C: Lightweight Mask Head (mini Cascade R-CNN)
RepViT backbone → SimpleFPN → RPN → ROI pooling → mask head

Training: DropLoss + box loss + mask loss (CUPS recipe)
Inference: NMS → mask prediction per detection
- Extra params: ~2-5M (RPN + box head + mask head)
- Advantage: Full CUPS compatibility, highest PQ_things ceiling
- Disadvantage: Heavy, complex, needs Detectron2, defeats "mobile" purpose

My recommendation

Option B (Center + Offset) is the sweet spot for mobile:

- Only +0.15M params (total 5.05M, still well under 8M budget)
- No clustering at inference — just argmax on heatmap + offset lookup, runs in <1ms
- The offset vectors naturally separate touching objects (each pixel points to its center, not the neighbor's)
- Compatible with your pseudo-instance labels (generate center heatmaps + offset maps from depth-guided instances)
- Used in real-time panoptic systems deployed on mobile

The training pipeline would be:
1. Pre-compute center heatmaps + offset maps from pseudo_instance_spidepth/ labels
2. Add center head (1-ch) + offset head (2-ch) to SimpleFPN
3. Train jointly: CE (semantic) + MSE (centers) + L1 (offsets)
4. At inference: semantic map + center peaks + offset grouping → panoptic map