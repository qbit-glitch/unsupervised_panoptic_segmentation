#!/bin/bash                                                                                                
#SBATCH --partition=TODO                     # Request partition
#SBATCH --account=TODO
#SBATCH -J explora_dino                      # Job name
#SBATCH -o outputs/explora_dino%j.out        # output file (%j expands to jobID)
#SBATCH -N 1                                 # Total number of nodes requested
#SBATCH -n 8                                 # Total number of cores requested
#SBATCH --get-user-env                       # retrieve the users login environment
#SBATCH --mem=90000                          # server memory requested (per node)
#SBATCH -t 120:00:00                         # Time limit (hh:mm:ss)
#SBATCH --gres=gpu:4                         # Type/number of GPUs needed
#SBATCH --constraint=16G                     # gpu memory


# If not using SLURM, set num_gpus manually
if [ -n "$SLURM_JOB_GPUS" ]; then
  num_gpus=$(echo $SLURM_JOB_GPUS | tr ',' '\n' | wc -l)
else
  num_gpus=4
fi

# list out some useful information (optional)
echo "SLURM_JOBID="$SLURM_JOBID
echo "SLURM_JOB_NODELIST"=$SLURM_JOB_NODELIST
echo "SLURM_NNODES"=$SLURM_NNODES
echo "SLURM_GPUS"=$SLURM_JOB_GPUS

echo "working directory = "$SLURM_SUBMIT_DIR

master_port=45001;

base_csv_dir="data_and_checkpoints/fmow_csvs" ;
base_experiment_dir="data_and_checkpoints/explora_dino" ;

# cfg_file="dinov2/configs/train/fmow_vitb14.yaml" ;
cfg_file="dinov2/configs/train/fmow_vitl14.yaml" ;
# cfg_file="dinov2/configs/train/fmow_vitg14.yaml" ;


out_dir="${base_experiment_dir}/dino-explora_imagenet-weights_rgb-pretrain_sdino_clsblk23r64-bs1024lr0.001-fle3"


export PYTHONPATH=.
WANDB__SERVICE_WAIT=300 torchrun --nproc_per_node=$num_gpus --master_port=$master_port dinov2/train/train.py \
    --wandb=dino_train \
    --config-file="${cfg_file}" \
    --output_dir="${out_dir}" \
    lora.unfreeze_blocks="[23]" lora.rank=64 lora.unfreeze_cls_token="true" \
    optim.base_lr=0.001 optim.freeze_last_layer_epochs=3 optim.accum_iter=32 \
    train.num_workers=8 train.batch_size_per_gpu=8
