# ============================
# Run LoRA Inference
# Load trained LoRA parameters and run inference
# ============================

import os
import json 
import torch
from tqdm import tqdm
from omegaconf import OmegaConf
from peft import PeftModel, PeftConfig
from nlp_models import get_general_model
from set_seed import set_seed 

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_version")
parser.add_argument("--lm_name")
parser.add_argument("--lm_size")
parser.add_argument("--lm_cache_dir")
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int)
parser.add_argument("--lora_weights_path", type=str, help="Path to the trained LoRA weights")
parser.add_argument("--prompt_version", type=str, default='general')

args = parser.parse_args()
flags = OmegaConf.create(vars(args))


# == Seed ==
set_seed(flags.seed)
checkpoint = flags.lora_weights_path.split('/')[-1]
flags.save_dir = f"outputs/lora_inference/{flags.data_version}/" \
                + f"{flags.lm_name}_{flags.lm_size}/prompt_version_{flags.prompt_version}/seed_{flags.seed}"

if not os.path.exists(flags.save_dir):
    os.makedirs(flags.save_dir)
OmegaConf.save(flags, os.path.join(flags.save_dir, 'config.yaml'))

# == Load base model ==
llm_config = {
    'lm_name': flags.lm_name,
    'lm_size': flags.lm_size,
    'lm_cache_dir': flags.lm_cache_dir,
    'device_map': flags.device_map
}
base_model, tokenizer = get_general_model(**llm_config)

# == Load LoRA configuration and model ==
# peft_config = PeftConfig.from_pretrained(flags.lora_weights_path)
model = PeftModel.from_pretrained(base_model, flags.lora_weights_path)
model.eval()

from lora_dataset import get_lora_dataset
train_dataloader, test_dataloader = get_lora_dataset(flags.data_version, 
                                                    flags.prompt_version, 
                                                    tokenizer, 
                                                    flags.batch_size, 
                                                    flags.seed)

# == Run inference ==
all_labels = []
all_predictions = []

from stopping_criteria import StoppingCriteriaSub
from transformers import StoppingCriteriaList
stops = ['<stop>']
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stops)])

pbar = tqdm(test_dataloader, total=len(test_dataloader))
pbar.set_description("Running inference with LoRA")
max_new_tokens = 20

with torch.no_grad():
    for j ,batch in enumerate(pbar):
        batch = {k: v.to(model.device) for k, v in batch.items()}
        input_ids = batch['input_ids']
        
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=batch['attention_mask'],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_return_sequences=1,
            pad_token_id=tokenizer.eos_token_id,
            stopping_criteria=stopping_criteria,
        )
        
        for i, output in enumerate(outputs):
            prediction = tokenizer.decode(output[input_ids.shape[-1]:], skip_special_tokens=True)
            all_predictions.append(prediction)
            
            # Print some examples
            if i == 0:  # Print first example from each batch
                print("\nExample prediction:")
                print(f"Prediction: {prediction[-20:]}")
                print("-" * 50)

        if j % 50 == 0:
            results = [{'prediction': prediction} for (prediction) in zip(all_predictions)]
            results_path = os.path.join(flags.save_dir, f'inference_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"\nResults saved to {results_path}")

# Save results
results = [{'prediction': prediction} for (prediction) in zip(all_predictions)]
results_path = os.path.join(flags.save_dir, 'inference_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {results_path}")