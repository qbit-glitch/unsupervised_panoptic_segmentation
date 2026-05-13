# MBPS NeurIPS 2026 Reproducibility Release

This directory is a clean paper-release copy for **Monocular Bootstrapping for Unsupervised Panoptic Segmentation**. It was assembled from the working MBPS codebase without modifying the original files.

The release covers the paper pipeline:

1. Stage 1 pseudo-label generation from monocular RGB:
   frozen DINOv2 + CAUSE-TR 90D codes, DCFA, k=80 clustering, DepthPro depth connected components, and SIMCF.
2. Stage 2 panoptic bootstrapping:
   CUPS Cascade Mask R-CNN with a frozen DINOv3 ViT-B/16 backbone.
3. Stage 3 EMA self-training:
   CUPS self-training on the same pseudo-label source.
4. Evaluation:
   Cityscapes 27-class CUPS protocol plus KITTI, Waymo, Mapillary, MOTS, and COCO-Stuff-27 checks.

## Directory Layout

```text
mbps_pytorch/              MBPS Stage-1 code and analysis utilities
repro_scripts/             copied experiment and preprocessing scripts
third_party/cups/          copied CUPS trainer/evaluator source, without experiments/checkpoints
third_party/cause/         copied CAUSE source modules needed for frozen 90D codes
third_party/dinov3/        copied DINOv3 package source
configs/                   copied MBPS configs
paper/                     paper TeX sources used to map claims to code
figures/paper_ready/       paper figures
paper_artifacts/           small result JSONs and reports used by paper tables
tests/test_release_smoke.py release-level smoke and integrity tests
```

Large artifacts are intentionally not included: Cityscapes data, DepthPro maps, DINO/CAUSE/DINOv3 weights, Lightning checkpoints, pseudo-label PNG directories, W&B logs, and CUPS experiment folders.

For this machine, the original absolute locations of those large assets are recorded in `LOCAL_ASSET_PATHS.md`.

## Environment

Use Python 3.10. A typical setup is:

```bash
cd mbps_neurips2026_repro
python3.10 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements-release.txt
pip install -r third_party/cups/requirements.txt
pip install -e third_party/dinov3
```

Some CUPS dependencies, especially Detectron2, are platform specific. Install the Detectron2 build matching your PyTorch and CUDA versions before running Stage 2 or Stage 3.

## Required Local Artifacts

On this workstation, use the paths in `LOCAL_ASSET_PATHS.md`. For a portable/public run, keep the following layout and set the same variables to your own local copies.

Expected data layout:

```text
$CITYSCAPES_ROOT/
  leftImg8bit/{train,val}/...
  gtFine/{train,val}/...
  depth_depthpro/{train,val}/...       # DepthPro .npy maps
  dinov3_features/train/...            # stride-16 DINOv3 features for SIMCF Step B
```

Expected weights:

```text
third_party/cause/checkpoint/dinov2_vit_base_14.pth
third_party/cause/CAUSE/cityscapes/dinov2_vit_base_14/2048/segment_tr.pth
third_party/cause/CAUSE/cityscapes/modularity/dinov2_vit_base_14/2048/modular.npy
weights/dinov3_vitb16_official.pth
```

Stage-3 evaluation also needs the trained Lightning checkpoint:

```text
checkpoints/stage3_dcfa_simcf_abc/best_pq_step=003000.ckpt
```

## End-to-End Commands

Set paths first:

```bash
export CITYSCAPES_ROOT=/path/to/cityscapes
export PYTHONPATH="$PWD:$PWD/third_party/cause:$PWD/third_party/cups:$PWD/third_party/dinov3:${PYTHONPATH:-}"
```

Extract frozen CAUSE codes:

```bash
python repro_scripts/extract_cause_codes.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --output_subdir cause_codes_90d
```

Train DCFA:

```bash
python mbps_pytorch/train_depth_adapter.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --codes_subdir cause_codes_90d \
  --output_dir results/depth_adapter/V3_dd16_h384_l2 \
  --adapter_type v3 \
  --depth_dim 16 \
  --hidden_dim 384 \
  --num_layers 2 \
  --lambda_preserve 20.0 \
  --lr 1e-3 \
  --epochs 20 \
  --seed 42
```

Generate DCFA k=80 semantic clusters:

```bash
python mbps_pytorch/generate_depth_overclustered_semantics.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --split train \
  --adapter_checkpoint results/depth_adapter/V3_dd16_h384_l2/best.pt \
  --codes_subdir cause_codes_90d \
  --depth_subdir depth_depthpro \
  --variant sinusoidal \
  --alpha 0.1 \
  --k 80 \
  --output_subdir pseudo_semantic_adapter_V3_k80 \
  --skip_crf \
  --raw_clusters
```

Convert to CUPS format with DepthPro instances:

```bash
python mbps_pytorch/convert_to_cups_format.py \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --semantic_subdir pseudo_semantic_adapter_V3_k80 \
  --output_subdir cups_pseudo_labels_adapter_V3_tau020 \
  --split train \
  --num_classes 80 \
  --depth_cc_instances \
  --centroids_path "$CITYSCAPES_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz" \
  --depth_subdir depth_depthpro \
  --grad_threshold 0.20 \
  --depth_blur_sigma 0.0 \
  --dilation_iters 3 \
  --min_instance_area 1000
```

Apply SIMCF:

```bash
python repro_scripts/refine_simcf.py \
  --input_dir "$CITYSCAPES_ROOT/cups_pseudo_labels_adapter_V3_tau020" \
  --output_dir "$CITYSCAPES_ROOT/cups_pseudo_labels_dcfa_simcf_abc" \
  --centroids_path "$CITYSCAPES_ROOT/pseudo_semantic_adapter_V3_k80/kmeans_centroids.npz" \
  --cityscapes_root "$CITYSCAPES_ROOT" \
  --steps A,B,C \
  --features_subdir dinov3_features \
  --depth_subdir depth_depthpro \
  --sim_threshold 0.85 \
  --sigma_threshold 2.5 \
  --num_clusters 80
```

Run CUPS Stage 2:

```bash
cd third_party/cups
python -u train.py \
  --experiment_config_file configs/train_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb
```

Run CUPS Stage 3:

```bash
cd third_party/cups
python -u train_self.py \
  --experiment_config_file configs/train_self_cityscapes_dinov3_vitb_dcfa_simcf_abc_santosh.yaml \
  --disable_wandb
```

Evaluate:

```bash
cd third_party/cups
python -u val.py \
  --experiment_config_file configs/val_stage3_dcfa_simcf_abc_local.yaml
```

The copied configs still contain machine-specific absolute paths from the original experiments. Before a public run, edit only your release copy or override config keys on the command line.

## Tests

Run release smoke tests:

```bash
python -m pytest tests/test_release_smoke.py -q
```

Run a syntax pass:

```bash
python -m compileall -q mbps_pytorch repro_scripts third_party/cups/cups third_party/cause third_party/dinov3/dinov3
```

These tests verify the copied source tree, DCFA identity initialization and parameter count, SIMCF toy behavior, CUPS config parsing, and presence of the result artifacts behind the paper tables. They do not run full Cityscapes training because that requires external data, weights, and GPUs.
