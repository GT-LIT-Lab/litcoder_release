#!/bin/bash
#SBATCH --account=gts-aivanova7-lab
#SBATCH -N1 --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH -t 12:00:00
#SBATCH -qinferno
#SBATCH -o jobreports/Report-%A%a.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=tbinhuraib3@gatech.edu

cd $SLURM_SUBMIT_DIR
module load anaconda3
conda activate /storage/coda1/p-aivanova7/0/shared/sae_llms

CACHE_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/cache"
DATA_DIR="/storage/coda1/p-aivanova7/0/shared/taha/SAE_encoding_models/lit_encoding/data/lebel/neural_data"

# Run for fullcontext only
python $SLURM_SUBMIT_DIR/../train_lebel_speech.py \
    --data_dir $DATA_DIR \
    --subject UTS03 \
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
    --wandb_project_name litcoderpublic_lebel_speech_test_release \