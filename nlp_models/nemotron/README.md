## NVIDIA Nemotron

nvidia/Llama-3.1-Nemotron-70B-Instruct-HF

```python
I ate bread because I was feeling blue. What do you think?

# Llama-3.1-Nemotron-70B-Instruct-HF 70B
Is that a good reason to eat bread?
I think it's a common reason to eat bread, but not necessarily a good
```

### Structure

```python
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 8192)
    (layers): ModuleList(
      (0-79): 80 x LlamaDecoderLayer(
        (self_attn): LlamaSdpaAttention(
          (q_proj): Linear(in_features=8192, out_features=8192, bias=False)
          (k_proj): Linear(in_features=8192, out_features=1024, bias=False)
          (v_proj): Linear(in_features=8192, out_features=1024, bias=False)
          (o_proj): Linear(in_features=8192, out_features=8192, bias=False)
          (rotary_emb): LlamaRotaryEmbedding()
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=8192, out_features=28672, bias=False)
          (up_proj): Linear(in_features=8192, out_features=28672, bias=False)
          (down_proj): Linear(in_features=28672, out_features=8192, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((8192,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((8192,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=8192, out_features=128256, bias=False)
)

```

### Configuration

```python
# LLAMA3_1_INSTRUCT
LlamaConfig {
  "_name_or_path": "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 128000,
  "eos_token_id": [
    128001,
    128008,
    128009
  ],
  "head_dim": 128,
  "hidden_act": "silu",
  "hidden_size": 8192,
  "initializer_range": 0.02,
  "intermediate_size": 28672,
  "max_position_embeddings": 131072,
  "mlp_bias": false,
  "model_type": "llama",
  "num_attention_heads": 64,
  "num_hidden_layers": 80,
  "num_key_value_heads": 8,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": {
    "factor": 8.0,
    "high_freq_factor": 4.0,
    "low_freq_factor": 1.0,
    "original_max_position_embeddings": 8192,
    "rope_type": "llama3"
  },
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 128256
}
```

----

## Token Examples

```bash 
====================
9891 yes
9642 Yes
14331 YES
====================
20137 :YES
14331 YES
77259 ĠReyes
53522 ĠHayes
99234 ĠBayesian
124814 ĠBelediyesi
13985 Ġyesterday
9891 yes
14410 ĠYES
10035 Ġyes
7566 ĠYes
56170 Ġpolyester
75949 ĠPolyester
61133 ĠYesterday
95934 =YES
58841 "Yes
86508 âĢľYes
60844 _yes
77994 .YesNo
60665 =yes
43096 ĠEyes
112850 Ġsayesinde
41898 .Yes
85502 .YES
98171 ,Yes
77830 _YES
51377 Yesterday
9642 Yes
58331 eyes
6548 Ġeyes
114767 iyesi
```