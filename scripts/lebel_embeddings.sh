#!/bin/bash

# Set up directories
CACHE_DIR="cache"
DATA_DIR="data/lebel/neural_data"

# Define array of subject IDs
SUBJECTS=(
    "UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07"
)

# Define embedding models and their paths
declare -A EMBEDDING_PATHS=(
    ["word2vec"]="GoogleNews-vectors-negative300.bin.gz"
    ["glove"]="wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
)

# Activate conda environment
conda activate litcoder

for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"
    
    # Run for each embedding model
    for MODEL in "${!EMBEDDING_PATHS[@]}"; do
        echo "  Model: $MODEL"
        
        python ../train_lebel_embeddings.py \
            --data_dir $DATA_DIR \
            --subject $SUBJECT \
            --model_name $MODEL \
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
            --vector_path ${EMBEDDING_PATHS[$MODEL]} \
            --wandb_project_name litcoderpublic_lebel_embeddings_test
    done
done
