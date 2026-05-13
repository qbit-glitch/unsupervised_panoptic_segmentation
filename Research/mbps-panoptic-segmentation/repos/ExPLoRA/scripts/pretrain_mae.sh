#!/bin/bash                                                                                                
#SBATCH --partition=TODO                     # Request partition
#SBATCH --account=TODO
#SBATCH -J explora_mae                       # Job name
#SBATCH -o outputs/explora_mae%j.out         # output file (%j expands to jobID)
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

base_dir="data_and_checkpoints" ;
base_csv_dir="${base_dir}/fmow_csvs" ;
base_experiment_dir="${base_dir}/explora_mae" ;

dataset_type="rgb" ;
train_csv="${base_csv_dir}/train_62classes.csv"
val_csv="${base_csv_dir}/test_62classes.csv"

lora_type="lora";
lora_rank=64 ;
decoder_lora_rank=8 ;

model_type="vanilla" ;
input_size=224 ;
patch_size=16 ;
batch_size=64 ;
accum_iter=4 ;
epochs=400 ;
blr=4.5e-4 ;

pretrain_ckpt="${base_dir}/mae_visualize_vit_large.pth"

out_dir="${base_experiment_dir}/${lora_type}_in-weights_${dataset_type}-pretrain_blk23_r${lora_rank}_dr${decoder_lora_rank}-bs1024"

torchrun --nproc_per_node=$num_gpus --master_port=49999 mae/pretrain.py \
    --wandb=fmow_lora_pretrain \
    --output_dir="${out_dir}" \
    --log_dir="${out_dir}" \
    --seed=42 \
    --batch_size="${batch_size}" --accum_iter="${accum_iter}" \
    --epochs="${epochs}" --warmup_epochs 1 \
    --blr="${blr}" --mask_ratio 0.75 \
    --input_size="${input_size}" --patch_size="${patch_size}" \
    --model mae_vit_large_patch16 \
    --norm_pix_loss \
    --lora_type="${lora_type}" --lora_rank="${lora_rank}" --decoder_lora_rank="${decoder_lora_rank}" \
    --unfreeze_norm --unfreeze_cls_token --unfreeze_blocks 23 --lora_layers attn  \
    --model_type="${model_type}" \
    --load_weights="${pretrain_ckpt}" \
    --num_workers 8 --dataset_type="${dataset_type}" \
    --train_path="${train_csv}" \
##    --resume="${out_dir}/checkpoint-15.pth" 