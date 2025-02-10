import torch
from transformers import StoppingCriteria
class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, tokenizer, stops=[]):
        super().__init__()
        self.stops = stops
        self.tokenizer = tokenizer
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs):
        num_last = 10
        input_ids = input_ids[:, -num_last:]
        dones = []
        for i in range(input_ids.shape[0]):
            found = False
            for stop in self.stops:
                if stop in self.tokenizer.decode(input_ids[i]):
                    dones.append(True)
                    found = True 
                    break
            if not found:
                dones.append(False)
        return torch.tensor(dones, dtype=torch.bool, device=input_ids.device)  # torch.BoolTensor of shape (B,)
