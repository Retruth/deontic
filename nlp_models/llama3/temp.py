from nlp_models.llama3.load_model import get_llama3

lm_cache_dir = '/data1/bumjin/datahub'
model, tokenizer = get_llama3('llama3_instruct', '70b', lm_cache_dir, device_map='cpu')
print(model)