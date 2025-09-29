#!/bin/bash

# Set up directories
CACHE_DIR="cache"
DATA_DIR="data/lebel/neural_data"

SUBJECTS=("UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07")

# Activate conda environment
conda activate litcoder

# Iterate over number of delays from 1 to 9
for NDELAYS in {1..9}; do
    echo "Processing with ndelays: $NDELAYS"
    
    PYTHONPATH=/Users/tahabinhuraib/Documents/litcoder_public/lit_encoding python ..train_lebel.py \
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

