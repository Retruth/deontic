#!/bin/bash

# Base directories
DATA_CACHE_DIR="/data1/bumjin/datahub"
LM_CACHE_DIR="/data1/bumjin/datahub"
BATCH_SIZE=4

# Array of models to test
declare -a models=(
    # "llama3_instruct 8b"
    # "exaone 8b"
    "gemma2 9b"
    # "gemma2 9b"
    # "llama2_hf 7b"
    # "llama2_chat_hf 7b"
    # "qwen2 7b"
)

# Seeds for multiple runs
declare -a seeds=(42)
declare -a datasets=("truthful_qa")  # "imdb"
declare -a prompt_types=(
    # "zero_shot_cot" 
    "bayesian"
    "markov"
    # "base"
)
label_types=("self_correction" )  # "self_correction"
checkpoints=(3 5 10 20) 
for model in "${models[@]}"; do
    # Split model name and size
    read -r lm_name lm_size <<< "$model"
    for label_type in "${label_types[@]}"; do
    for seed in "${seeds[@]}"; do
        for dataset in "${datasets[@]}"; do
            for prompt_type in "${prompt_types[@]}"; do
                for checkepoint in "${checkpoints[@]}"; do 
                echo "Running with model: $lm_name ($lm_size) - seed: $seed - dataset: $dataset - prompt_type: $prompt_type"
                LORA_WEIGHTS_PATH='outputs/lora_finetuning/'$dataset'/'$prompt_type'/'$lm_name'_'$lm_size'/'$label_type'/seed_'$seed'/checkpoint-epoch-'$checkepoint
                python scripts/run_lora_inference.py \
                --prompt_type "$prompt_type" \
                --dataset_name "$dataset" \
                --data_cache_dir "$DATA_CACHE_DIR" \
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
done
