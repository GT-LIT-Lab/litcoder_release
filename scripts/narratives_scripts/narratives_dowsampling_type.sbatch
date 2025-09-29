#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 06:00:00
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu
#SBATCH --array=0-12

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache_narratives"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/narratives/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "sub-249" "sub-255" "sub-256" "sub-257" "sub-258" "sub-260" "sub-261" "sub-263" "sub-264" "sub-265" "sub-267" "sub-269" "sub-262"
)

# Get current subject based on array task ID
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Array of downsampling methods to test
METHODS=("legacy_average" "legacy_last" "lanczos")

for METHOD in "${METHODS[@]}"; do
    python $SLURM_SUBMIT_DIR/../../training_files/train_narratives.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --model_name gpt2-small \
        --layer_idx 8 \
        --last_token \
        --folding_type kfold_trimmed \
        --chunk_length 20 \
        --singcutoff 1e-10 \
        --downsample_method $METHOD \
        --lanczos_cutoff_mult 1.0 \
        --lanczos_window 3 \
        --lookback 256 \
        --cache_dir $CACHE_DIR \
        --normalize_features \
        --normalize_targets \
        --ndelays 8 \
        --tr 1.5 \
        --context_type fullcontext \
        --wandb_project_name "narratives_downsampling_type_final"
done
