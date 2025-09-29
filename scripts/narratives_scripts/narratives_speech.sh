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
    
    python ../../train_narratives_speech.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --speech_model_name facebook/hubert-base-ls960 \
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
        --ndelays 8 \
        --tr 1.5 \
        --wandb_project_name "litcoderpublic_narratives_test_speech" \
        --wav_path "21styear_audio.wav"
    
    echo "Completed training for subject $SUBJECT"
done

