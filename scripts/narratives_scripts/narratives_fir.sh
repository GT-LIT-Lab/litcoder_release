#!/bin/bash

# Set up directories
CACHE_DIR="cache_narratives"
DATA_DIR="data/narratives/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "sub-244" "sub-249" "sub-254" "sub-255" "sub-256" "sub-257" "sub-258" "sub-259"
    "sub-260" "sub-261" "sub-263" "sub-264" "sub-265" "sub-266" "sub-267"
    "sub-268" "sub-269"
)

# Activate conda environment
conda activate litcoder

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Starting training for subject $SUBJECT"
    
    # Loop over different FIR delays
    for NDELAYS in {1..12}; do
        echo "  Training with $NDELAYS delays"
        
        python ../../train_narratives.py \
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
done
