#!/bin/bash

# Base directories
DATA_CACHE_DIR="/data1/bumjin/datahub"
LM_CACHE_DIR="/data1/bumjin/datahub"

# Training hyperparameters
BATCH_SIZE=2
LEARNING_RATE=1e-4
WEIGHT_DECAY=0.01
NUM_EPOCHS=20
WARMUP_RATIO=0.1
GRADIENT_ACCUMULATION_STEPS=8
MAX_GRAD_NORM=0.5 # 1.0 for others 

# LoRA hyperparameters
LORA_R=8
LORA_ALPHA=16
LORA_DROPOUT=0.05

# Array of models to test
declare -a models=(
    "llama3_instruct 8b"
    "gemma2 9b"
    "exaone 8b"
    # "llama2_hf 7b"
    # "llama2_chat_hf 7b"
    # "gemma1 7b"
    # "qwen2 7b"
    # "llama2_hf 13b"
    # "llama2_chat_hf 13b"
)

# Seeds for multiple runs
declare -a lora_label_types=("self_correction")  # "self_correction" 
declare -a seeds=(42)
declare -a datasets=("truthful_qa")
declare -a prompt_types=("zero_shot_cot" "bayesian" "markov" "base" ) #)   # "zero_shot_cot"  "base" 

for model in "${models[@]}"; do
    # Split model name and size
    read -r lm_name lm_size <<< "$model"
    
    for seed in "${seeds[@]}"; do
        for dataset in "${datasets[@]}"; do
            for prompt_type in "${prompt_types[@]}"; do
                for lora_label_type in "${lora_label_types[@]}"; do 
                    echo "----------------------------------------"
                    echo "Running LoRA fine-tuning with:"
                    echo "- Model: $lm_name ($lm_size)"
                    echo "- Seed: $seed"
                    echo "- Dataset: $dataset"
                    echo "- Prompt type: $prompt_type"
                    echo "- LoRA label type: $lora_label_type"
                    echo "----------------------------------------"

                python scripts/run_lora_finetuning.py \
                --prompt_type "$prompt_type" \
                --dataset_name "$dataset" \
                --data_cache_dir "$DATA_CACHE_DIR" \
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
                --lora_label_type "$lora_label_type"
                done
            done
        done
    done
done
