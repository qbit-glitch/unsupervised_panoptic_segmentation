# Local Asset Path Reference

This file records the original machine paths used while assembling this release. These paths are intentionally absolute because the large datasets, checkpoints, feature caches, and experiment folders are not copied into the GitHub release directory.

Use this file as a local lookup table. For a public release, keep the generic instructions in `README.md` as the portable entry point and treat this file as machine-specific provenance.

## Source Tree Origins

| Release path | Original path on this machine |
| --- | --- |
| `mbps_pytorch/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_pytorch/` |
| `repro_scripts/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/scripts/` |
| `third_party/cups/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups/` |
| `third_party/cause/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/` |
| `third_party/dinov3/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/dinov3/` |
| `paper/mbps_neurips2026_mbps.tex` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/paper/mbps_neurips2026_mbps.tex` |
| `paper/mbps_neurips2026_supplementary.tex` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/paper/mbps_neurips2026_supplementary.tex` |
| `figures/paper_ready/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/figures/paper_ready/` |
| `paper_artifacts/results/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/results/`, `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cups/results/`, and selected root eval JSONs |
| `paper_artifacts/reports/` | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/reports/` |

## Local Runtimes

| Purpose | Path |
| --- | --- |
| Original project root | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/` |
| Release root | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_neurips2026_repro/` |
| Python used for release smoke tests | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/.venv/bin/python` |
| Python 3.10 environment used by older CUPS scripts | `/Users/qbit-glitch/Desktop/datasets/.venv_py310/bin/python` |

## Cityscapes Paper Assets

Set `CITYSCAPES_ROOT` to:

```bash
export CITYSCAPES_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
```

| Asset | Original path | Current local check |
| --- | --- | --- |
| Cityscapes root | `/Users/qbit-glitch/Desktop/datasets/cityscapes/` | 113G total |
| Train RGB images | `/Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit/train/` | 2975 `*_leftImg8bit.png` files |
| Val RGB images | `/Users/qbit-glitch/Desktop/datasets/cityscapes/leftImg8bit/val/` | 500 `*_leftImg8bit.png` files |
| Cityscapes GT | `/Users/qbit-glitch/Desktop/datasets/cityscapes/gtFine/` | present |
| DepthPro depth maps | `/Users/qbit-glitch/Desktop/datasets/cityscapes/depth_depthpro/` | 3476 files |
| Frozen CAUSE 90D code cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/cause_codes_90d/` | 13900 files |
| DINOv2 feature cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov2_features/` | present |
| DINOv3 ViT-B/16 feature cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features/` | 3478 files |
| DINOv3 ViT-L/16 feature cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features_vitl16/` | present |
| High-res DINOv3 feature cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/dinov3_features_hires/` | present |

## Stage 1 Pseudo-Label Assets

| Asset | Original path | Current local check |
| --- | --- | --- |
| DCFA k=80 semantic labels | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_adapter_V3_k80/` | 3476 files |
| DCFA k=80 centroids | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz` | present |
| DepthPro CUPS labels before SIMCF | `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_adapter_V3_tau020/` | 8925 files |
| Final DCFA + SIMCF-ABC CUPS labels used by paper configs | `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/` | 8925 files |
| DA3/DCFA/SIMCF-ABC alternate pseudo-label folder | `/Users/qbit-glitch/Desktop/datasets/cityscapes/cups_pseudo_labels_dcfa_da3_simcf_abc/` | 8925 files |
| Final pseudo-label eval cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_eval_cups_pseudo_labels_dcfa_simcf_abc/` | present |
| SIMCF-ABC eval cache | `/Users/qbit-glitch/Desktop/datasets/cityscapes/pseudo_eval_cups_pseudo_labels_simcf_abc/` | present |

## Weights And Checkpoints

| Asset | Original path | Current local check |
| --- | --- | --- |
| DINOv2 ViT-B/14 CAUSE backbone | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/checkpoint/dinov2_vit_base_14.pth` | 330M |
| CAUSE Segment-TR checkpoint | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth` | 70M |
| CAUSE Cluster-TR checkpoint | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/cluster_tr.pth` | 6.0M |
| CAUSE modular codebook | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048/modular.npy` | 6.0M |
| DINOv3 ViT-B/16 official weights | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/dinov3_vitb16_official.pth` | 327M |
| DINOv3 ViT-B/16 HF state dict | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/dinov3_vitb16_hf_statedict.pth` | 327M |
| Alternate DINOv3 HF state dict copy | `/Users/qbit-glitch/Desktop/datasets/dinov3_vitb16_hf_statedict.pth` | 327M |
| Original CUPS checkpoint | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/cups.ckpt` | 916M |
| Paper Stage-3 final checkpoint | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt` | 1.4G |
| Other local Stage-3 checkpoints | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/checkpoints/stage3_dcfa_simcf_abc/` | 5.5G total |

## Transfer Evaluation Data Roots

| Dataset | Original path | Current local check |
| --- | --- | --- |
| COCO-Stuff-27 / COCO assets | `/Users/qbit-glitch/Desktop/datasets/coco/` | 82G |
| KITTI panoptic | `/Users/qbit-glitch/Desktop/datasets/kitti_panoptic/` | 845M |
| Mapillary Vistas v2 | `/Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2/` | 30G |
| MOTSChallenge | `/Users/qbit-glitch/Desktop/datasets/MOTSChallenge/` | 468M |
| Waymo V2 preprocessed path from eval config | `/Volumes/code_files/datasets/panoptic_segmentation_datasets/waymo_v2_0_1_preprocessed/` | missing on this local mount during this check |
| Empty local Waymo placeholder | `/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/code_files/datasets/panoptic_segmentation_dataset/waymo_v2_0_1/` | 0B |

## Remote GPU Training References

These paths come from the copied experiment configs and prior run scripts. They are remote training references, not copied release assets.

| Purpose | Remote reference |
| --- | --- |
| Stage-2 target noted in config | `santosh@100.93.203.100`, 2x GTX 1080 Ti |
| Stage-3 target noted in config | `santosh@172.17.254.146`, 2x GTX 1080 Ti |
| Remote Cityscapes root in Stage-2/3 configs | `/home/santosh/datasets/cityscapes/` |
| Remote final pseudo-label root in Stage-2/3 configs | `/home/santosh/datasets/cityscapes/cups_pseudo_labels_dcfa_simcf_abc/` |
| Remote Stage-2 log root in config | `/home/santosh/experiments/stage2_dcfa_simcf_abc/` |
| Remote Stage-3 log root in config | `/home/santosh/experiments/stage3_dcfa_simcf_abc/` |
| Remote Stage-2 checkpoint loaded by Stage-3 config | `/home/santosh/experiments/stage2_dcfa_simcf_abc/experiments/cups_dinov3_vitb_dcfa_simcf_abc_2gpu/Unsupervised Panoptic Segmentation/v70uy7wv/checkpoints/best_pq_step=000744.ckpt` |

## Quick Local Exports

```bash
export MBPS_ORIGINAL=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation
export MBPS_RELEASE=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/mbps_neurips2026_repro
export CITYSCAPES_ROOT=/Users/qbit-glitch/Desktop/datasets/cityscapes
export COCO_ROOT=/Users/qbit-glitch/Desktop/datasets/coco
export KITTI_PANOPTIC_ROOT=/Users/qbit-glitch/Desktop/datasets/kitti_panoptic
export MAPILLARY_ROOT=/Users/qbit-glitch/Desktop/datasets/mapillary-vistas-v2
export MOTS_ROOT=/Users/qbit-glitch/Desktop/datasets/MOTSChallenge
export DINOV3_VITB16_WEIGHTS=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/weights/dinov3_vitb16_official.pth
export CAUSE_DINOV2_WEIGHTS=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/checkpoint/dinov2_vit_base_14.pth
export CAUSE_SEGMENT_TR=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/refs/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth
export MBPS_STAGE3_CKPT=/Users/qbit-glitch/Desktop/coding-projects/mbps_panoptic_segmentation/checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt
```
