from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def get_qwq(lm_name, lm_size, lm_cache_dir, device_map='auto', precision=None, return_tokenizer_only=False):

    model_name = "Qwen/QwQ-32B-Preview"
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=lm_cache_dir)

    if return_tokenizer_only:
        return tokenizer
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16 if precision is None else precision,
        device_map=device_map ,
        cache_dir=lm_cache_dir,
    )
    tokenizer.sep_token_id =  tokenizer.eos_token_id 
    tokenizer.pad_token_id =  tokenizer.eos_token_id 
    return model, tokenizer


    