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

from procllm.data.truthful_qa.prepare import prepare_truthful_qa, get_prompt_fn
from procllm.nlp_models import get_general_model
from procllm.utils.set_seed import set_seed

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--data_cache_dir")
parser.add_argument("--prompt_type")
parser.add_argument("--dataset_name")
parser.add_argument("--lm_name")
parser.add_argument("--lm_size")
parser.add_argument("--lm_cache_dir")
parser.add_argument("--device_map", type=str, default="auto")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seed", type=int)
parser.add_argument("--lora_weights_path", type=str, help="Path to the trained LoRA weights")

args = parser.parse_args()
flags = OmegaConf.create(vars(args))


label_type = args.lora_weights_path.split('/')[-3]
print(label_type)

# == Seed ==
set_seed(flags.seed)
checkpoint = flags.lora_weights_path.split('/')[-1]
flags.save_dir = f"outputs/lora_inference/{flags.dataset_name}/{flags.prompt_type}/" \
                + f"{flags.lm_name}_{flags.lm_size}/{label_type}/{checkpoint}/seed_{flags.seed}"

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

# == Prepare data ==
if flags.dataset_name == "truthful_qa":
    prompt_fn = get_prompt_fn(flags.prompt_type)
    _, test_dataloader = prepare_truthful_qa(
        flags.data_cache_dir, 
        tokenizer, 
        flags.batch_size, 
        flags.seed, 
        prompt_fn
    )
else:
    raise ValueError(f"Invalid dataset name: {flags.dataset_name}")

# == Run inference ==
all_labels = []
all_predictions = []

from procllm.utils.stopping_criteria import StoppingCriteriaSub
from transformers import StoppingCriteriaList
stops = ['<stop>']
stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(tokenizer, stops=stops)])

pbar = tqdm(test_dataloader, total=len(test_dataloader))
pbar.set_description("Running inference with LoRA")
max_new_tokens = {
    'base': 50,
    'zero_shot_cot': 500,
    'bayesian': 500,    
    'markov': 500,
    'simple_markov': 500,
    'simple_bayesian': 500,
}[flags.prompt_type]

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
            label = tokenizer.decode(batch['labels'][i], skip_special_tokens=True)
            all_predictions.append(prediction)
            all_labels.append(label)
            
            # Print some examples
            if i == 0:  # Print first example from each batch
                print("\nExample prediction:")
                print(f"Label: {label}")
                print(f"Prediction: {prediction[-20:]}")
                print("-" * 50)

        if j % 50 == 0:
            results = [{'label': label, 'prediction': prediction} for (label, prediction) in zip(all_labels, all_predictions)]
            results_path = os.path.join(flags.save_dir, f'inference_results.json')
            with open(results_path, 'w') as f:
                json.dump(results, f, indent=4)

            print(f"\nResults saved to {results_path}")

# Save results
results = [{'label': label, 'prediction': prediction} for (label, prediction) in zip(all_labels, all_predictions)]
results_path = os.path.join(flags.save_dir, 'inference_results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"\nResults saved to {results_path}")