import torch
from transformers import GPTNeoXForCausalLM, AutoTokenizer

PYTHIA_MODEL_SIZES_LAYERS = {
    '70m':6, 
    '160m':12, 
    '410m':24, 
    '1b':16, 
    '1.4b':24, 
    '2.8b':32, 
    '6.9b':32, 
    '12b':36
}

def get_pythia(lm_name, lm_size, lm_cache_dir, device_map='auto', precision=None, return_tokenizer_only=False):
    lm_size = lm_size.lower()
    if lm_name == "pythia":
        name = f'EleutherAI/pythia-{lm_size}-deduped'
    else:
        raise ValueError(f"Invalid model name: {lm_name}")

    if return_tokenizer_only:
        tokenizer = AutoTokenizer.from_pretrained(name,
                                                  cache_dir=None if lm_cache_dir in ['none', None] else lm_cache_dir)
        return tokenizer
    
    if isinstance(device_map, int):
        device_map =custom_device_map(lm_size, device_map)
    precision = torch.bfloat16 if precision is None else precision
    model = GPTNeoXForCausalLM.from_pretrained(
            name,
            revision='step143000',
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
    model_size = PYTHIA_MODEL_SIZES_LAYERS[lm_size]
    device_map = {'gpt_neox.embed_in':0, 
                  'gpt_neox.final_layer':num_gpus-1,
                  'gpt_neox.final_layer_norm':num_gpus-1,
                  'embed_out': num_gpus-1,
                  }
    gpu = 0
    per_block = model_size//num_gpus
    for i in range(1, model_size+1):
        device_map[f'gpt_neox.layers.{i-1}'] = gpu
        if i % per_block ==0 and  gpu < num_gpus-1:
            gpu += 1
    return device_map   
