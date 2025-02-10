def get_general_model(lm_name, lm_size, lm_cache_dir, device_map='auto', precision=None, return_tokenizer_only=False):
    from .llama3.load_model import get_llama3
    from .exaone.load_model import get_exaone
    from .llama2.load_model import get_llama2
    from .gemma2.load_model import get_gemma2
    from .gemma1.load_model import get_gemma1
    from .qwen2.load_model import get_qwen2
    from .pythia.load_model import get_pythia
    from .nemotron.load_model import get_nemotron
    from .openai.load_model import OpenAIClient
    from .qwq.load_model import get_qwq 
    
    if lm_name[:len("llama3")] == "llama3":
        return get_llama3(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("exaone")] == "exaone":
        return get_exaone(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("llama2")] == "llama2":
        return get_llama2(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("openai")] == "openai":
        return OpenAIClient(lm_name, lm_size), None
    elif lm_name[:len("gemma2")] == "gemma2":
        return get_gemma2(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("gemma1")] == "gemma1":
        return get_gemma1(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("qwen2")] == "qwen2":
        return get_qwen2(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("pythia")] == "pythia":
        return get_pythia(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("nemotron")] == "nemotron":
        return get_nemotron(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)
    elif lm_name[:len("qwq")] == "qwq":
        return get_qwq(lm_name, lm_size, lm_cache_dir, device_map, precision, return_tokenizer_only)    
    else:
        raise ValueError(f"Invalid model name: {lm_name}")
