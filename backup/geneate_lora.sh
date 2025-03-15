#!/bin/bash

# Base directories
DATA_CACHE_DIR="/data1/bumjin/datahub"
LM_CACHE_DIR="/data1/bumjin/datahub"
BATCH_SIZE=4

# Array of models to test
declare -a models=(
    # "gemma2 9b"
    "llama3_instruct 8b"
    "qwen2 7b"
    # "exaone 8b"
    # "gemma2 9b"
    # "gemma2 9b"
    # "llama2_hf 7b"
    # "llama2_chat_hf 7b"
    # "qwen2 7b"
)

# Seeds for multiple runs
declare -a seeds=(0)
checkpoints=(10 20 30) 
for model in "${models[@]}"; do
    # Split model name and size
    read -r lm_name lm_size <<< "$model"
    for prompt_version in general explicit strict; do
        for data_version in 1 2 3 4; do
            for seed in "${seeds[@]}"; do
                for checkpoint in "${checkpoints[@]}"; do 
                    echo "Running with model: $lm_name ($lm_size) - seed: $seed - checkpoint: $checkpoint"
                    LORA_WEIGHTS_PATH='outputs/lora_finetuning/'$data_version'/'$lm_name'_'$lm_size'/prompt_version_'$prompt_version'/seed_'$seed'/checkpoint-epoch-'$checkpoint
                    python run_lora_inference.py \
                    --data_version "$data_version" \
                    --prompt_version "$prompt_version" \
                    --lm_name "$lm_name" \
                    --lm_size "$lm_size" \
                    --lm_cache_dir "$LM_CACHE_DIR" \
                    --batch_size "$BATCH_SIZE" \
                    --seed "$seed" \
                    --lora_weights_path "$LORA_WEIGHTS_PATH"
                done 
            done 
        done 
    done 
done 
