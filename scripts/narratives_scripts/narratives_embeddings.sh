#!/bin/bash

# Set up directories
CACHE_DIR="cache_narratives"
DATA_DIR="data/narratives/neural_data"
VECTOR_DIR="vector_embeddings"

# Define array of subject IDs
SUBJECTS=(
    "sub-244" "sub-249" "sub-254" "sub-255" "sub-256" "sub-257" "sub-258" "sub-259"
    "sub-260" "sub-261" "sub-263" "sub-264" "sub-265" "sub-266" "sub-267"
    "sub-268" "sub-269"
)

# Define array of embedding models and their paths
declare -A EMBEDDINGS=(
    ["word2vec"]="GoogleNews-vectors-negative300.bin.gz"
    ["glove"]="glove.840B.300d.txt"
)

# Activate conda environment
conda activate litcoder

# Loop over all subjects and embedding types
for SUBJECT in "${SUBJECTS[@]}"; do
    echo "Processing subject: $SUBJECT"
    
    for MODEL in "${!EMBEDDINGS[@]}"; do
        echo "  Embedding model: $MODEL"
        
        VECTOR_PATH="${VECTOR_DIR}/${EMBEDDINGS[$MODEL]}"
        
        python ../../train_narratives_embeddings.py \
            --data_dir $DATA_DIR \
            --subject $SUBJECT \
            --model_name $MODEL \
            --layer_idx 7 \
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
            --context_type fullcontext \
            --vector_path $VECTOR_PATH \
            --wandb_project_name "litcoderpublic_narratives_test_embeddings"
    done
done
