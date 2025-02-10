#!/bin/bash

# Base directories
LM_CACHE_DIR="/data1/bumjin/datahub"

# Training hyperparameters
BATCH_SIZE=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
NUM_EPOCHS=30
WARMUP_RATIO=0.1
GRADIENT_ACCUMULATION_STEPS=4
MAX_GRAD_NORM=0.5 # 1.0 for others 

# LoRA hyperparameters
LORA_R=16
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Array of models to test
declare -a models=(
    "gemma2 9b"
    "llama3_instruct 8b"
    "qwen2 7b"
    # "exaone 8b"
    # "llama2_hf 7b"
    # "llama2_chat_hf 7b"
    # "gemma1 7b"
    # "llama2_hf 13b"
    # "llama2_chat_hf 13b"
)


# Seeds for multiple runs
for model in "${models[@]}"; do
    # Split model name and size
    read -r lm_name lm_size <<< "$model"
    for prompt_version in general explicit strict; do
        for data_version in 1 2 3 4; do
            for seed in 0; do
                echo "----------------------------------------"
                echo "Running LoRA fine-tuning with:"
                echo "- Model: $lm_name ($lm_size)"
                echo "- Seed: $seed"
                echo "----------------------------------------"

                python run_lora_finetuning.py \
                --data_version "$data_version" \
                --lm_name "$lm_name" \
                --lm_size "$lm_size" \
                --lm_cache_dir "$LM_CACHE_DIR" \
                --batch_size "$BATCH_SIZE" \
                --seed "$seed" \
                --learning_rate "$LEARNING_RATE" \
                --weight_decay "$WEIGHT_DECAY" \
                --num_epochs "$NUM_EPOCHS" \
                --warmup_ratio "$WARMUP_RATIO" \
                --gradient_accumulation_steps "$GRADIENT_ACCUMULATION_STEPS" \
                --max_grad_norm "$MAX_GRAD_NORM" \
                --lora_r "$LORA_R" \
                --lora_alpha "$LORA_ALPHA" \
                --lora_dropout "$LORA_DROPOUT" \
                --prompt_version "$prompt_version"
            done
        done
    done
done
