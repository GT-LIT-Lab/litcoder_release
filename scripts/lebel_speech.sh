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

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing speech model for subject: $SUBJECT"
    
    python ../train_lebel_speech.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --model_name openai/whisper-tiny \
        --layer_idx 3 \
        --last_token \
        --folding_type chunked \
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
        --wandb_project_name litcoderpublic_lebel_speech_test
done

echo "All subjects processed!"