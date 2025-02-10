
from openai import OpenAI

class OpenAIClient:
    def __init__(self, lm_name, lm_size,  **kwargs):
        self.client = OpenAI()
        self.lm_name = lm_name
        self.lm_size = lm_size
        
    def generate(self, 
                 prompt, 
                 max_tokens, 
                 temperature=0.0,
                 top_p=1.0,
                 **kwargs):
        response = self.client.chat.completions.create(
                        model=self.lm_size,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        top_p=top_p,
                    )
        prediction = response.choices[0].message.content.strip()
        return prediction
    
    