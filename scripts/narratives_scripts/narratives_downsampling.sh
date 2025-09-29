#!/bin/bash

# Set up directories
CACHE_DIR="cache_narratives"
DATA_DIR="data/narratives/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "sub-249" "sub-255" "sub-256" "sub-257" "sub-258" "sub-260" "sub-261" "sub-263" "sub-264" "sub-265" "sub-267" "sub-269" "sub-262"
)

# Array of downsampling methods to test
METHODS=("legacy_average" "legacy_last" "lanczos")

# Activate conda environment
conda activate litcoder

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"
    
    # Loop through each downsampling method
    for METHOD in "${METHODS[@]}"; do
        echo "  Downsampling method: $METHOD"
        
        python ../../training_files/train_narratives.py \
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
done
