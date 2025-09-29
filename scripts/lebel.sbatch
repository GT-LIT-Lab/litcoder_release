#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-gpu=256GB
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu
#SBATCH --array=0-11

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache2"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/lebel"

# Define array of subject IDs
SUBJECTS=(
    "UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07"
)

# Current layer based on array task ID
LAYER=$SLURM_ARRAY_TASK_ID

# Run for each subject
for SUBJECT in "${SUBJECTS[@]}"; do
    python $SLURM_SUBMIT_DIR/../train_lebel.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --model_name gpt2-small \
        --layer_idx $LAYER \
        --last_token \
        --folding_type kfold \
        --chunk_length 20 \
        --singcutoff 1e-10 \
        --downsample_method lanczos \
        --lanczos_cutoff_mult 1.0 \
        --lanczos_window 3 \
        --trim_start 50 \
        --trim_end 5 \
        --lookback 128 \
        --cache_dir $CACHE_DIR \
        --context_type fullcontext \
        --wandb_project_name litcoderpublic_lebel_test_2 \
done