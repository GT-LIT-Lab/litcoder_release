#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu
#SBATCH --array=0-0

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/lebel/neural_data"

# Define array of subject IDs
# SUBJECTS=(
#     "UTS08" "UTS01" "UTS02" "UTS03" "UTS04" "UTS05" "UTS06" "UTS07"
# )
SUBJECTS=(
    "UTS03"
)

# Define embedding models and their paths
declare -A EMBEDDING_PATHS=(
    ["word2vec"]="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/GoogleNews-vectors-negative300.bin.gz"
    ["glove"]="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/wiki_giga_2024_300_MFT20_vectors_seed_2024_alpha_0.75_eta_0.05_combined.txt"
)

# Get current subject based on array task ID
SUBJECT=${SUBJECTS[$SLURM_ARRAY_TASK_ID]}

# Run for each embedding model
for MODEL in "${!EMBEDDING_PATHS[@]}"; do
    python $SLURM_SUBMIT_DIR/../train_lebel_embeddings.py \
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
        --wandb_project_name litcoderpublic_lebel_embeddings_test_release
done