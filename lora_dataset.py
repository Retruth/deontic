import datasets 
from torch.utils.data import DataLoader
from transformers import default_data_collator
def get_tokenized_dataset(dataset, tokenizer, batch_size, num_proc=None, 
                          target_name='text', 
                          output_name='input_ids',
                          preserve_names=[],
                          max_length=None,
                          padding=True,
                          truncation=False):
    def process(samples):
        batch_inputs = {}    
        tokenized = tokenizer(samples[target_name], 
                                max_length=max_length, 
                                padding=padding, 
                                truncation=truncation, 
                                return_tensors='pt')     
        
        batch_inputs[output_name] = tokenized['input_ids']
        batch_inputs['attention_mask'] = tokenized['attention_mask']
        
        for v in preserve_names:
            batch_inputs[v] = samples[v]
        
        if 'label' in samples and isinstance(samples['label'][0], str):
            batch_inputs['label_ids'] = tokenizer(samples['label'], 
                                            max_length=max_length, 
                                            padding=padding, 
                                            truncation=truncation, 
                                            return_tensors='pt')['input_ids']
        return batch_inputs
    
    tokenizer.padding_side = "left"
    remove_columns = dataset.column_names
    dataset = dataset.map(  
                process,
                batched=True,
                num_proc=num_proc,
                load_from_cache_file=False,
                desc="Tokenizing dataset...",
                batch_size=batch_size,
                remove_columns= remove_columns
        )
    return dataset

def get_dataloder_from_dataset(dataset, batch_size, shuffle=False):
    dataloader = DataLoader(dataset, batch_size=batch_size, 
                                            shuffle=shuffle, 
                                            collate_fn=default_data_collator,
                                            pin_memory=True)
    return dataloader

def get_lora_dataset(data_cache_dir, tokenizer, batch_size, seed, stopping_word = "<stop>"):
    prompt = "The answer to the question is:"
    train_dic = {
        'text': [ prompt + " " + str(x) + " " + stopping_word for x in range(1000) ],
    }
    test_dic = {
        'text': [ prompt + " " + str(x) + " " + stopping_word for x in range(1000) ],
    }
    
    train_dataset = datasets.Dataset.from_dict(train_dic)
    test_dataset = datasets.Dataset.from_dict(test_dic)
    
    tokenized_train = get_tokenized_dataset(train_dataset, tokenizer, batch_size)
    tokenized_test = get_tokenized_dataset(test_dataset, tokenizer, batch_size)
    train_dataloader = get_dataloder_from_dataset(tokenized_train, batch_size)
    test_dataloader = get_dataloder_from_dataset(tokenized_test, batch_size)
    
    return train_dataloader, test_dataloader