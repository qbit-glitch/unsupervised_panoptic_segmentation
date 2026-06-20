#!/bin/bash
# GA-DepthG-UNet ablation: unet+geom (geometry skips) vs unet (no skip). Baseline simple-head = 14.43.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/HDD_16TB/umesh/envs/gadepthg
cd /mnt/HDD_16TB/umesh/ga_depthg/depthg
export WANDB_MODE=disabled
RL=/mnt/HDD_16TB/umesh/ga_depthg/runlogs
mkdir -p "$RL"
COMMON="data_dir=/mnt/HDD_16TB/umesh/ga_depthg/data output_root=/mnt/HDD_16TB/umesh/ga_depthg/outputs dataset_name=cityscapes model_type=vit_base dim=100 arch=dino_unet crop_type=null res=224 use_depth=True depth_type=geometry depth_feat_correlation_loss=False guidance=none depth_sampling=none pointwise=False batch_size=32 max_steps=2000 val_freq=1000 checkpoint_freq=2000 num_workers=12 +submitting_to_aml=False +azureml_logging=False"
for spec in "geom True" "nogeom False"; do
  set -- $spec; NAME=$1; SKIP=$2
  echo "[$(date)] START cs_unet_$NAME (unet_geom_skip=$SKIP)"
  python src/train_segmentation.py $COMMON unet_geom_skip=$SKIP experiment_name=cs_unet_$NAME > "$RL/cs_unet_$NAME.log" 2>&1
  echo "[$(date)] DONE cs_unet_$NAME -> $(grep -oE 'test/cluster/mIoU[^,}]*' "$RL/cs_unet_$NAME.log" | tail -1)"
done
echo "[$(date)] ALL UNET RUNS DONE"
