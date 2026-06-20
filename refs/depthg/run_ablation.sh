#!/bin/bash
# GA-DepthG Stage-A 4-way ablation (res=224, 2000 steps). scalar+both first (kill-gate), then height/normal.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate /mnt/HDD_16TB/umesh/envs/gadepthg
cd /mnt/HDD_16TB/umesh/ga_depthg/depthg
export WANDB_MODE=disabled
RL=/mnt/HDD_16TB/umesh/ga_depthg/runlogs
mkdir -p "$RL"
COMMON="data_dir=/mnt/HDD_16TB/umesh/ga_depthg/data output_root=/mnt/HDD_16TB/umesh/ga_depthg/outputs dataset_name=cityscapes model_type=vit_base dim=100 arch=dino crop_type=null res=224 use_depth=True depth_feat_correlation_loss=True guidance=none depth_sampling=none pointwise=False batch_size=32 max_steps=2000 num_workers=12 val_freq=1000 checkpoint_freq=2000 depth_feat_weight=0.09 depth_feat_shift=0.03 depth_loss_decay=True depth_loss_decay_factor=0.8 decay_every_steps=400 pos_intra_weight=0.95 pos_intra_shift=0.39 pos_inter_weight=1.02 pos_inter_shift=0.25 neg_inter_weight=0.57 neg_inter_shift=0.26 +submitting_to_aml=False +azureml_logging=False"
for spec in "scalar depthpro" "both geometry" "height geometry" "normal geometry"; do
  set -- $spec; MODE=$1; DT=$2
  echo "[$(date)] START cs_$MODE (affinity=$MODE depth=$DT)"
  python src/train_segmentation.py $COMMON depth_type=$DT affinity_mode=$MODE experiment_name=cs_$MODE > "$RL/cs_$MODE.log" 2>&1
  echo "[$(date)] DONE cs_$MODE -> $(grep -oE 'test/cluster/mIoU[^,}]*' "$RL/cs_$MODE.log" | tail -1)"
done
echo "[$(date)] ALL 4 RUNS DONE"
