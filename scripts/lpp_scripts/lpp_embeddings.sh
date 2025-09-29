#!/bin/bash

# Set up directories
CACHE_DIR="cache_lpp"
DATA_DIR="data/little_prince/neural_data"

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

# Activate conda environment
conda activate litcoder

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"
    
    python ../../train_lpp_embeddings.py \
        --data_dir $DATA_DIR \
        --subject $SUBJECT \
        --model_name word2vec \
        --layer_idx 3 \
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
        --vector_path GoogleNews-vectors-negative300.bin.gz \
        --wandb_project_name litcoderpublic_lpp_test_embeddings
done
