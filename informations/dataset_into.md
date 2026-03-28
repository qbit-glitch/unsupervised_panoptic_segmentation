Here's a comprehensive breakdown of both papers' dataset usage:

  ---
  Paper 1: CUPS — Scene-Centric Unsupervised Panoptic Segmentation (Hahn et al., CVPR 2025)

  Core Training Dataset

  - Cityscapes (training sequences) — stereo video frames used to generate pseudo labels (via depth + motion cues) and train the panoptic network. This is the primary
  dataset the model is built on.

  Evaluation Datasets

  ┌───────────────┬────────────────────────────────────────────────┬──────────────────────────────────────────────────────────────────────────────────────────────────┐
  │    Dataset    │                      Role                      │                                             Details                                              │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Cityscapes    │ Primary evaluation                             │ 19 semantic categories, PQ/SQ/RQ metrics                                                         │
  │ val           │                                                │                                                                                                  │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ KITTI         │ Cross-domain generalization                    │ Trained on Cityscapes, tested on KITTI                                                           │
  │ panoptic      │                                                │                                                                                                  │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ BDD100K       │ Cross-domain generalization                    │ Driving scenes, domain shift from Cityscapes                                                     │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MUSES         │ Cross-domain generalization                    │ Multi-sensor driving dataset                                                                     │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ Waymo         │ Cross-domain generalization + instance seg     │ Also used for class-agnostic instance segmentation benchmark (Table 4)                           │
  │               │ eval                                           │                                                                                                  │
  ├───────────────┼────────────────────────────────────────────────┼──────────────────────────────────────────────────────────────────────────────────────────────────┤
  │ MOTS          │ Out-of-domain (OOD) evaluation                 │ Multi-object tracking & segmentation — tests generalization to a completely different task       │
  │               │                                                │ domain                                                                                           │
  └───────────────┴────────────────────────────────────────────────┴──────────────────────────────────────────────────────────────────────────────────────────────────┘

  Ablation-Specific Dataset Usage

  - Cityscapes val — All ablations (Tables 5, 6, 7a-c) are conducted on Cityscapes val pseudo labels
  - KITTI-raw — Table 7d ablates using KITTI as an alternative training dataset (instead of Cityscapes), validated on KITTI panoptic. Demonstrates dataset versatility.

  Datasets for Sub-Task Comparisons

  - Cityscapes — unsupervised semantic segmentation (Table 3, 27 classes, mIoU + Acc)
  - Waymo — unsupervised instance segmentation (Table 4, mask AP metrics)

  ---
  Paper 2: U2Seg — Unsupervised Universal Image Segmentation (Niu et al., CVPR 2024)

  Core Training Dataset

  - ImageNet (1.3M unlabeled images) — the primary dataset used to train the model in the zero-shot setting. No labels used.
  - For non-zero-shot scenarios: COCO (unlabeled) or COCO + ImageNet (unlabeled mix)
  - For semantic segmentation comparisons with STEGO: additionally fine-tuned on MSCOCO's unlabeled images

  Evaluation Datasets

  ┌────────────────────┬──────────────────────────┬─────────────────────────────────────────────────────┐
  │      Dataset       │           Task           │                       Details                       │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ COCO val2017       │ Instance seg (zero-shot) │ AP^box, AR^box metrics                              │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ COCO val2017       │ Panoptic seg             │ PQ, SQ, RQ (Table 6)                                │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ COCO val2017       │ Object detection         │ AP^box, AR^box (Table 2)                            │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ PASCAL VOC val2012 │ Instance seg (zero-shot) │ AP^box, AR^box (Table 3)                            │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ UVO val            │ Instance seg (zero-shot) │ AP^box, AR^box (Table 4) — open-world video objects │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ COCOStuff          │ Semantic segmentation    │ PixelAcc, mIoU (Table 1)                            │
  ├────────────────────┼──────────────────────────┼─────────────────────────────────────────────────────┤
  │ Cityscapes val     │ Panoptic segmentation    │ PQ, SQ, RQ (Table 5) — non-zero-shot                │
  └────────────────────┴──────────────────────────┴─────────────────────────────────────────────────────┘

  Ablation Datasets

  - COCO, UVO, VOC — Number of clusters ablation (Table 7): tested 300 / 800 / 2911 clusters across all three instance seg datasets
  - COCO val2017 — Hungarian matching parameter ablation (Table 8): confidence threshold and IoU threshold sensitivity

  Label-Efficient Learning Datasets

  - COCO — 1% and 2% labeled data fine-tuning for object detection, instance seg, and panoptic seg (Figure 7)

  ---
  Side-by-Side Summary

  ┌───────────────────────┬───────────────────────────────────────────────────────────┬────────────────────────────────────────────────────────────┐
  │        Aspect         │                        CUPS (2025)                        │                        U2Seg (2024)                        │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Core training data    │ Cityscapes stereo sequences (scene-centric)               │ ImageNet 1.3M (object-centric)                             │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Training signal       │ Depth + motion + SSL features from stereo video           │ MaskCut + DINO + K-means clustering                        │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Primary eval          │ Cityscapes val (panoptic)                                 │ COCO val2017 (panoptic + instance)                         │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Cross-domain eval     │ KITTI, BDD, MUSES, Waymo, MOTS                            │ PASCAL VOC, UVO, Cityscapes, COCOStuff                     │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Total unique datasets │ 7 (Cityscapes, KITTI, BDD, MUSES, Waymo, MOTS, KITTI-raw) │ 6 (ImageNet, COCO, PASCAL VOC, UVO, COCOStuff, Cityscapes) │
  ├───────────────────────┼───────────────────────────────────────────────────────────┼────────────────────────────────────────────────────────────┤
  │ Key distinction       │ Scene-centric (driving/outdoor), leverages stereo video   │ Object-centric, works from static images                   │
  └───────────────────────┴───────────────────────────────────────────────────────────┴────────────────────────────────────────────────────────────┘

  The fundamental difference: CUPS builds its model around Cityscapes stereo video (exploiting depth and motion as free supervision), while U2Seg builds on ImageNet static
   images (exploiting SSL features and clustering). CUPS then shows it generalizes to other driving datasets, while U2Seg shows it generalizes across diverse
  object-centric benchmarks.
