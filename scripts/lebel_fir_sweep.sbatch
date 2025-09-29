#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 06:00:00
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/lebel/neural_data"

SUBJECTS=("UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07")

# Iterate over number of delays from 1 to 9
for NDELAYS in {1..9}; do
    PYTHONPATH=/Users/tahabinhuraib/Documents/litcoder_public/lit_encoding python $SLURM_SUBMIT_DIR/../training_files/train_lebel.py \
        --data_dir $DATA_DIR \
        --subject UTS03 \
        --model_name gpt2-small \
        --layer_idx 6 \
        --last_token \
        --folding_type chunked \
        --chunk_length 20 \
        --singcutoff 1e-10 \
        --ndelays $NDELAYS \
        --downsample_method lanczos \
        --lanczos_cutoff_mult 1.0 \
        --lanczos_window 3 \
        --trim_start 50 \
        --trim_end 5 \
        --lookback 128 \
        --cache_dir $CACHE_DIR \
        --context_type fullcontext \
        --wandb_project_name fir_test_litcoderpublic
done