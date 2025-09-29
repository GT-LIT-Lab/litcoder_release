#!/bin/bash

# Set up directories
CACHE_DIR="cache"
DATA_DIR="data/lebel/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07"
)

# Activate conda environment
conda activate litcoder

for LAYER in {0..11}; do
    echo "Processing layer $LAYER"
    
    # Run for each subject
    for SUBJECT in "${SUBJECTS[@]}"; do
        echo "  Subject: $SUBJECT"
        
        python ../train_lebel.py \
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
            --wandb_project_name litcoderpublic_lebel_test_2
    done
done
