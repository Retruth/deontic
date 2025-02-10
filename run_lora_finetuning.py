# ============================
# Run LoRA Finetuning
# Using PEFT for efficient training
# ============================

import os
import json 
import torch
import numpy as np
from tqdm import tqdm
import pickle
from omegaconf import OmegaConf
from lora import LoRaWrapper
from procllm.utils.set_seed import set_seed

import argparse
import matplotlib.pyplot as plt
from torch.cuda.amp import GradScaler, autocast
from transformers import get_cosine_schedule_with_warmup

parser = argparse.ArgumentParser()
parser.add_argument("--data_version")
parser.add_argument("--lm_name")
parser.add_argument("--lm_size")
parser.add_argument("--lm_cache_dir")
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int)
# Training hyperparameters
parser.add_argument("--learning_rate", type=float, default=3e-4)
parser.add_argument("--weight_decay", type=float, default=0.01)
parser.add_argument("--num_epochs", type=int, default=3)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
parser.add_argument("--max_grad_norm", type=float, default=1.0)
# LoRA specific arguments
parser.add_argument("--lora_r", type=int, default=8)
parser.add_argument("--lora_alpha", type=int, default=16)
parser.add_argument("--lora_dropout", type=float, default=0.05)
parser.add_argument("--prompt_version", type=str, default='general')
args = parser.parse_args()
flags = OmegaConf.create(vars(args))

# == Seed ==
set_seed(flags.seed)
flags.save_dir = f"outputs/lora_finetuning/{flags.data_version}/" \
                + f"{flags.lm_name}_{flags.lm_size}/seed_{flags.seed}"

if not os.path.exists(flags.save_dir):
    os.makedirs(flags.save_dir)
if os.path.exists(os.path.join(flags.save_dir, 'checkpoint-epoch-final')):
    print(f"Model already exists at {flags.save_dir}")
    exit()
    
OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml'))

# == make data & llm ==
# ========================================================================================
llm_config = {
    'lm_name': flags.lm_name,
    'lm_size': flags.lm_size,
    'lm_cache_dir': flags.lm_cache_dir,
    'device_map': flags.device_map
}
from nlp_models import get_general_model
llm, tokenizer = get_general_model(**llm_config)
# ========================================================================================
from lora_dataset import get_lora_dataset
train_dataloader, test_dataloader = get_lora_dataset(flags.data_version, 
                                                    flags.prompt_version, 
                                                    tokenizer, 
                                                    flags.batch_size, 
                                                    flags.seed)

# == Prepare model for LoRA ==
model = LoRaWrapper(llm, tokenizer)
model.prepare_model_for_kbit_training()
model.adapt_lora()
model.print_trainable_parameters()

# Example setup for optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=flags.learning_rate,
    betas=(0.9, 0.999),
    eps=1e-8,
    weight_decay=flags.weight_decay
)

# Add after optimizer creation
scaler = GradScaler()

# Add warmup to scheduler
num_training_steps = len(train_dataloader) * flags.num_epochs
num_warmup_steps = int(num_training_steps * flags.warmup_ratio)

# Replace the CosineAnnealingLR with a scheduler that supports warmup
scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
)

# Replace training loop with safer version
try:
    model.train()
    loss_values = []
    
    pbar = tqdm(total=num_training_steps, desc=flags.save_dir)

    for epoch in range(flags.num_epochs):
        total_loss = 0
        pbar.set_postfix({"epoch": epoch + 1})
        for batch_idx, batch in enumerate(train_dataloader):
            pbar.update(1)
            batch = {k: v.to(model.device) for k, v in batch.items()}
            batch['labels'] = batch['input_ids'].to(model.device)
            
            current_lr = optimizer.param_groups[0]['lr']  # Get current learning rate
            
            with autocast():
                outputs = model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    labels=batch["labels"],
                )
                loss = outputs.loss / flags.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % flags.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=flags.max_grad_norm)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
            
            total_loss += loss.item()
            pbar.set_postfix({"loss": loss.item(), "lr": current_lr})
            loss_values.append(loss.item())


        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} average loss: {avg_loss}")
        
        # Save checkpoint
        checkpoint_dir = os.path.join(flags.save_dir, f"checkpoint-epoch-{epoch + 1}")
        model.save_pretrained(checkpoint_dir)

    # Save final model
    model.save_pretrained(os.path.join(flags.save_dir, f"checkpoint-epoch-final"))

except Exception as e:
    print(f"Training failed with error: {str(e)}")
    raise
finally:
    # Cleanup
    model.remove_lora()
    torch.cuda.empty_cache()


# check lora load
model.load_lora(os.path.join(flags.save_dir, f"checkpoint-epoch-final"))
model.print_trainable_parameters()    

# Update loss plot saving
plt.figure(figsize=(10, 6))
plt.plot(loss_values)
pickle.dump(loss_values, open(os.path.join(flags.save_dir, 'training_loss.pkl'), 'wb'))
plt.title('Training Loss')
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.savefig(os.path.join(flags.save_dir, 'training_loss.png'))
plt.close()
