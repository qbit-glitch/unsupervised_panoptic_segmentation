Plan: Cross-Dataset Generalizability Evaluation on KITTI-STEP + Mapillary Vistas                                                                           
                                                                                                                                                        
Context                                                                                                                                                    
                                                                                                                                                        
We evaluated the Cityscapes-trained RepViT-M0.9+BiFPN model (PQ=24.78) on COCONUT and got PQ=0.9 — expected failure since 19 Cityscapes classes can't      
cover 133 COCO categories across diverse indoor/outdoor scenes. Now we want to test on same-domain (driving) datasets to see if the learned features       
actually generalize within the driving domain.                                                                                                             
                                                                                                                                                        
Script: mbps_pytorch/evaluate_cross_dataset.py                                                                                                             
                                                                                                                                                        
A single unified script that evaluates the model on both datasets, reusing the existing evaluate_mobile_on_coconut.py pattern (load model → inference →    
Hungarian matching → PQ eval).                                                                                                                             
                                                                                                                                                        
Part A: KITTI-STEP Evaluation                                                                                                                              
                                                                                                                                                        
Data: 2981 val images across 9 sequences, 1242×375 resolution                                                                                              

GT Format: 3-channel uint8 PNG panoptic maps
- B channel = semantic class (11=stuff, 12=?, 13=?) — need to map from KITTI-STEP's panoptic encoding
- R channel = instance ID
- Only stuff (0) and thing (1) semantic classes after remapping

Pipeline:
1. Load model, run inference on all val images from kitti-step/training/image_02/{seq_id}/ (val sequences: 0002, 0006, 0007, 0008, 0010, 0013, 0014, 0016,
0018)
2. Collapse 19 Cityscapes predictions → 2 classes: stuff (CS 0-10) vs thing (CS 11-18)
3. Generate panoptic maps via connected components on thing predictions
4. Evaluate against GT panoptic maps: PQ, PQ_stuff, PQ_things, SQ, RQ

Key files:
- Images: /Users/qbit-glitch/Desktop/datasets/kitti-step/training/image_02/{seq}/
- GT panoptic: /Users/qbit-glitch/Desktop/datasets/kitti-step/kitti-step/panoptic_maps/val/{seq}/
- CUPS reference: refs/cups/cups/data/kitti.py (KITTIPanopticValidation class)

Part B: Mapillary Vistas v2 Evaluation

Data: 2000 val images, variable resolution (up to 3840×2160)

GT Format:
- Semantic labels: uint8 PNG, label IDs 0-123 (124 classes)
- Panoptic maps: 3-channel uint8 PNG (COCO-style RGB encoding: R + G256 + B65536)
- Config: config_v2.0.json with class names, isthing flags

Strong Cityscapes ↔ Mapillary overlap (19 → 124 via Hungarian):

┌───────────────┬────────────────────────┬────────┐
│   CS Class    │    Mapillary Match     │   ID   │
├───────────────┼────────────────────────┼────────┤
│ road          │ Road                   │ 21     │
├───────────────┼────────────────────────┼────────┤
│ sidewalk      │ Sidewalk               │ 24     │
├───────────────┼────────────────────────┼────────┤
│ building      │ Building               │ 27     │
├───────────────┼────────────────────────┼────────┤
│ wall          │ Wall                   │ 12     │
├───────────────┼────────────────────────┼────────┤
│ fence         │ Fence                  │ 5      │
├───────────────┼────────────────────────┼────────┤
│ pole          │ Pole                   │ 85     │
├───────────────┼────────────────────────┼────────┤
│ traffic light │ Traffic Light variants │ 90-95  │
├───────────────┼────────────────────────┼────────┤
│ traffic sign  │ Traffic Sign variants  │ 96-103 │
├───────────────┼────────────────────────┼────────┤
│ vegetation    │ Vegetation             │ 64     │
├───────────────┼────────────────────────┼────────┤
│ terrain       │ Terrain                │ 63     │
├───────────────┼────────────────────────┼────────┤
│ sky           │ Sky                    │ 61     │
├───────────────┼────────────────────────┼────────┤
│ person        │ Person                 │ 30     │
├───────────────┼────────────────────────┼────────┤
│ rider         │ Bicyclist/Motorcyclist │ 32-34  │
├───────────────┼────────────────────────┼────────┤
│ car           │ Car                    │ 108    │
├───────────────┼────────────────────────┼────────┤
│ truck         │ Truck                  │ 114    │
├───────────────┼────────────────────────┼────────┤
│ bus           │ Bus                    │ 107    │
├───────────────┼────────────────────────┼────────┤
│ train         │ On-rails               │ 111    │
├───────────────┼────────────────────────┼────────┤
│ motorcycle    │ Motorcycle             │ 110    │
├───────────────┼────────────────────────┼────────┤
│ bicycle       │ Bicycle                │ 105    │
└───────────────┴────────────────────────┴────────┘

Pipeline:
1. Load model, run inference on 2000 val images
2. Save semantic predictions (19-class) as PNGs
3. Hungarian match 19 predictions → 124 Mapillary categories using semantic labels
4. Generate panoptic maps via connected components
5. Evaluate:
- Semantic mIoU (against v2.0/labels/)
- PQ (against v2.0/panoptic/ — requires loading panoptic JSON annotations)

GT annotation approach (no centralized JSON like COCO):
- For each image, load panoptic PNG + label PNG
- Derive segment category via majority vote of label pixels within each panoptic segment
- Build GT: {seg_id: {category_id, isthing, mask}}

Key files:
- Images: /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/validation/images/
- Semantic GT: /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/validation/v2.0/labels/ (uint8, IDs 0-123)
- Instance GT: /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/validation/v2.0/instances/ (uint16)
- Panoptic GT: /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/validation/v2.0/panoptic/ (RGB encoding)
- Config: /Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/config_v2.0.json (124 classes with isthing flags)

Implementation Steps

1. Check Mapillary panoptic JSON — verify panoptic.json or similar exists with segments_info
2. Create mbps_pytorch/evaluate_cross_dataset.py with:
- Shared: model loading, inference loop (reuse from evaluate_mobile_on_coconut.py)
- evaluate_kitti_step(): collapse to stuff/thing, CC instances, PQ vs GT
- evaluate_mapillary(): Hungarian matching 19→124, semantic mIoU + PQ
- CLI: --dataset kitti|mapillary|both
3. Run KITTI-STEP eval (~3000 images, ~2 min inference on MPS)
4. Run Mapillary eval (~2000 images, ~1.5 min inference, but larger images may need downscaling)

Reusable Code

- MobilePanopticModel from mbps_pytorch/train_mobile_panoptic.py
- _STUFF_IDS, _THING_IDS, _CS_CLASS_NAMES constants
- infer_instances_connected_components() from train_mobile_panoptic.py
- load_model() and inference pattern from evaluate_mobile_on_coconut.py
- Hungarian matching from evaluate_coconut_pseudolabels.py

Verification

1. KITTI-STEP: Check PQ on val set. Expect much better than COCONUT since same driving domain. Baseline: CUPS gets PQ=24.9 on KITTI-STEP.
2. Mapillary: Check mIoU and PQ. Expect decent mIoU on overlapping classes (road, sky, vegetation, car, person). Most of 19 CS classes should map
correctly.
3. Cross-compare: CS PQ=24.78 vs KITTI PQ vs Mapillary PQ to quantify domain gap.
