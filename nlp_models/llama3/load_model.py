import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


system_prompt = "A question from a human to an artificial intelligence assistant. " \
               + "The assistant gives helpful, detailed, and polite " \
               + "answers to the human's questions.\n"
user_prompt = "What is the capital of France?"
system_prompt_template = "[INST] <<SYS>>\n" \
                        + system_prompt \
                        + "\n<</SYS>>\n\n" \
                        + user_prompt


LLAMA_MODEL_SIZES_LAYERS ={
    '8b' : 32,
}

def get_llama3(lm_name, lm_size, lm_cache_dir, device_map='auto', precision=None, return_tokenizer_only=False):
    lm_size = lm_size.upper()
    if lm_name == "llama3_instruct":
        name = f"meta-llama/Meta-Llama-3-{lm_size}-Instruct"
    elif lm_name == "llama3":
        name = f"meta-llama/Meta-Llama-3-{lm_size}"
    elif lm_name == "llama3_1":
        name = f"meta-llama/Meta-Llama-3.1-{lm_size}"
    elif lm_name == "llama3_1_instruct":
        name = f"meta-llama/Meta-Llama-3.1-{lm_size}-Instruct"
    elif lm_name == "llama3_3_instruct":
        name = f"meta-llama/Llama-3.3-{lm_size}-Instruct"
    else:
        raise ValueError(f"Invalid model name: {lm_name}")

    if return_tokenizer_only:
        tokenizer = AutoTokenizer.from_pretrained(name,
                                                  cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir)
        return tokenizer
    
    if isinstance(device_map, int):
        device_map =custom_device_map(lm_size, device_map)
    precision = torch.bfloat16 if precision is None else precision
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=precision,
        trust_remote_code=True,
        cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir,
        device_map=device_map,
    )
    tokenizer = AutoTokenizer.from_pretrained(name,
                                              cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir )
    tokenizer.sep_token_id =  tokenizer.eos_token_id 
    tokenizer.pad_token_id =  tokenizer.eos_token_id 
    return model, tokenizer


def custom_device_map(lm_size, num_gpus):
    assert num_gpus > 0, "num_gpus must be greater than 0"
    model_size = LLAMA_MODEL_SIZES_LAYERS[model_size]
    device_map = {'model.embed_tokens':0, 
                  'model.norm':num_gpus-1,
                  'lm_head':num_gpus-1,
                  }
    gpu = 0
    per_block = model_size//num_gpus
    for i in range(1, model_size+1):
        device_map[f'model.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < num_gpus-1:
            gpu += 1
    return device_map   
