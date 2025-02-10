from peft import get_peft_model, PeftModel, LoraConfig, TaskType, prepare_model_for_kbit_training

class LoRaWrapper():
    def __init__(self, model, tokenizer, lora_r=8, lora_alpha=16, lora_dropout=0.1):
        self.model = model
        self.tokenizer = tokenizer
        
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=["q_proj", "v_proj"]  # Adjust based on your model architecture
        )
        
        self.lora_status = "original"
        
        
    def forward(self, input_ids, attention_mask, labels=None):
        # Ensure all inputs are on the same device
        if labels is not None:
            # Ensure labels are properly aligned with input
            labels = labels.to(input_ids.device)
            # Ensure labels have the correct shape
            if labels.shape != input_ids.shape:
                labels = labels.view(input_ids.shape)
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs


    def generate(self, input_ids, attention_mask, max_new_tokens=50, 
                 do_sample=False, num_return_sequences=1, pad_token_id=None, stopping_criteria=None):
        return self.model.generate(input_ids, attention_mask=attention_mask, max_new_tokens=max_new_tokens, 
                                 do_sample=do_sample, num_return_sequences=num_return_sequences, 
                                 pad_token_id=pad_token_id, stopping_criteria=stopping_criteria)

    def adapt_lora(self):
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()
        self.lora_status = "adapted"

    def load_lora(self, lora_config_path):
        if self.lora_status == "adapted":
            self.remove_lora()
            
        self.lora_config_path = lora_config_path
        self.model = get_peft_model(self.model, self.lora_config)
        self.model.print_trainable_parameters()

    def remove_lora(self):
        self.model = self.model.merge_and_unload()
        self.lora_status = "removed"

    def save_pretrained(self, save_path):
        self.model.save_pretrained(save_path)

    def prepare_model_for_kbit_training(self):
        self.model = prepare_model_for_kbit_training(self.model)

    def print_trainable_parameters(self):
        self.model.print_trainable_parameters()

    def parameters(self):
        return self.model.parameters()
    
    def to(self, device):
        self.model.to(device)

    @property
    def device(self):
        return next(self.model.parameters()).device

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()   