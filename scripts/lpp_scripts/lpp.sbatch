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
#SBATCH --array=0-44

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache_lpp2"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/little_prince/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "sub-EN068" 
    "sub-EN105" 
    "sub-EN095" 
    "sub-EN058" 
    "sub-EN103" 
    "sub-EN073" 
    "sub-EN065" 
    "sub-EN091" 
    "sub-EN081" 
    "sub-EN077" 
    "sub-EN098" 
    "sub-EN078" 
    "sub-EN086" 
    "sub-EN101" 
    "sub-EN067" 
    "sub-EN115" 
    "sub-EN106" 
    "sub-EN100" 
    "sub-EN076" 
    "sub-EN113" 
    "sub-EN096" 
    "sub-EN057" 
    "sub-EN070" 
    "sub-EN074" 
    "sub-EN072" 
    "sub-EN110" 
    "sub-EN089" 
    "sub-EN082" 
    "sub-EN104" 
    "sub-EN064" 
    "sub-EN109" 
    "sub-EN088" 
    "sub-EN108" 
    "sub-EN059" 
    "sub-EN063" 
    "sub-EN062" 
    "sub-EN114" 
    "sub-EN079"
    "sub-EN061" 
    "sub-EN083" 
    "sub-EN087" 
    "sub-EN084" 
    "sub-EN094" 
    "sub-EN092" 
    "sub-EN069"
)

SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

python $SLURM_SUBMIT_DIR/../../train_lpp_all.py \
    --data_dir $DATA_DIR \
    --subject $SUBJECT \
    --model_name gpt2-small \
    --layer_idx 11 \
    --last_token \
    --folding_type kfold_trimmed \
    --chunk_length 20 \
    --singcutoff 1e-10 \
    --downsample_method lanczos \
    --lanczos_cutoff_mult 1.0 \
    --lanczos_window 3 \
    --lookback 256 \
    --cache_dir $CACHE_DIR \
    --normalize_features \
    --normalize_targets \
    --ndelays 4 \
    --tr 2.0 \
    --context_type fullcontext \
    --wandb_project_name litcoderpublic_lpp_test_all_2
