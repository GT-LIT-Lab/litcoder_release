#!/bin/bash

# Set up directories
CACHE_DIR="cache"
DATA_DIR="data/lebel/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07"
)

# Define downsampling methods
DOWNSAMPLE_METHODS=("legacy_average" "legacy_last" "lanczos")

# Activate conda environment
conda activate litcoder

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"
    
    # Loop through each downsampling method
    for METHOD in "${DOWNSAMPLE_METHODS[@]}"; do
        echo "  Downsampling method: $METHOD"
        
        python ../training_files/train_lebel.py \
            --data_dir $DATA_DIR \
            --subject $SUBJECT \
            --model_name gpt2-small \
            --layer_idx 6 \
            --last_token \
            --folding_type kfold \
            --chunk_length 20 \
            --singcutoff 1e-10 \
            --ndelays 4 \
            --downsample_method $METHOD \
            --lanczos_cutoff_mult 1.0 \
            --lanczos_window 3 \
            --trim_start 50 \
            --trim_end 5 \
            --lookback 128 \
            --cache_dir $CACHE_DIR \
            --context_type fullcontext \
            --wandb_project_name lebel_downsampling_type_final
    done
done

echo "All subjects and downsampling methods processed!"