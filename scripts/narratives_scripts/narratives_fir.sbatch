#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 06:00:00
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu
#SBATCH --array=0-16

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache_narratives"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/narratives/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "sub-244" "sub-249" "sub-254" "sub-255" "sub-256" "sub-257" "sub-258" "sub-259"
    "sub-260" "sub-261" "sub-263" "sub-264" "sub-265" "sub-266" "sub-267"
    "sub-268" "sub-269"
)

# Get current subject based on array task ID
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

echo "Starting training for subject $SUBJECT"

# Loop over different FIR delays
for NDELAYS in {1..12}; do
    echo "Training with $NDELAYS delays"
    python $SLURM_SUBMIT_DIR/../../training_files/train_narratives.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --model_name gpt2-small \
        --layer_idx 8 \
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
        --ndelays $NDELAYS \
        --tr 1.5 \
        --context_type fullcontext \
        --wandb_project_name "fir_sweep_narratives_final"
done

echo "Completed training for subject $SUBJECT"