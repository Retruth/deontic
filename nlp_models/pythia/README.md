# Qwen2


```
I ate bread because I was feeling blue. What do you think?

I ate bread because I was feeling
```

## Model

```bash 

#6.9b
GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): Embedding(50432, 4096)
    (emb_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-31): 32 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
        (post_attention_dropout): Dropout(p=0.0, inplace=False)
        (post_mlp_dropout): Dropout(p=0.0, inplace=False)
        (attention): GPTNeoXSdpaAttention(
          (rotary_emb): GPTNeoXRotaryEmbedding()
          (query_key_value): Linear(in_features=4096, out_features=12288, bias=True)
          (dense): Linear(in_features=4096, out_features=4096, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): Linear(in_features=4096, out_features=16384, bias=True)
          (dense_4h_to_h): Linear(in_features=16384, out_features=4096, bias=True)
          (act): GELUActivation()
        )
      )
    )
    (final_layer_norm): LayerNorm((4096,), eps=1e-05, elementwise_affine=True)
    (rotary_emb): GPTNeoXRotaryEmbedding()
  )
  (embed_out): Linear(in_features=4096, out_features=50432, bias=False)
)


# 12b 
GPTNeoXForCausalLM(
  (gpt_neox): GPTNeoXModel(
    (embed_in): Embedding(50688, 5120)
    (emb_dropout): Dropout(p=0.0, inplace=False)
    (layers): ModuleList(
      (0-35): 36 x GPTNeoXLayer(
        (input_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        (post_attention_layernorm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
        (post_attention_dropout): Dropout(p=0.0, inplace=False)
        (post_mlp_dropout): Dropout(p=0.0, inplace=False)
        (attention): GPTNeoXSdpaAttention(
          (rotary_emb): GPTNeoXRotaryEmbedding()
          (query_key_value): Linear(in_features=5120, out_features=15360, bias=True)
          (dense): Linear(in_features=5120, out_features=5120, bias=True)
          (attention_dropout): Dropout(p=0.0, inplace=False)
        )
        (mlp): GPTNeoXMLP(
          (dense_h_to_4h): Linear(in_features=5120, out_features=20480, bias=True)
          (dense_4h_to_h): Linear(in_features=20480, out_features=5120, bias=True)
          (act): GELUActivation()
        )
      )
    )
    (final_layer_norm): LayerNorm((5120,), eps=1e-05, elementwise_affine=True)
    (rotary_emb): GPTNeoXRotaryEmbedding()
  )
  (embed_out): Linear(in_features=5120, out_features=50688, bias=False)
)
```

```bash
#6.9b
GPTNeoXConfig {
  "_name_or_path": "EleutherAI/pythia-6.9b-deduped",
  "architectures": [
    "GPTNeoXForCausalLM"
  ],
  "attention_bias": true,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "classifier_dropout": 0.1,
  "eos_token_id": 0,
  "hidden_act": "gelu",
  "hidden_dropout": 0.0,
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 16384,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neox",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "partial_rotary_factor": 0.25,
  "rope_scaling": null,
  "rope_theta": 10000,
  "rotary_emb_base": 10000,
  "rotary_pct": 0.25,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_parallel_residual": true,
  "vocab_size": 50432
}

# 12b 
GPTNeoXConfig {
  "_name_or_path": "EleutherAI/pythia-12b-deduped",
  "architectures": [
    "GPTNeoXForCausalLM"
  ],
  "attention_bias": true,
  "attention_dropout": 0.0,
  "bos_token_id": 0,
  "classifier_dropout": 0.1,
  "eos_token_id": 0,
  "hidden_act": "gelu",
  "hidden_dropout": 0.0,
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 20480,
  "layer_norm_eps": 1e-05,
  "max_position_embeddings": 2048,
  "model_type": "gpt_neox",
  "num_attention_heads": 40,
  "num_hidden_layers": 36,
  "partial_rotary_factor": 0.25,
  "rope_scaling": null,
  "rope_theta": 10000,
  "rotary_emb_base": 10000,
  "rotary_pct": 0.25,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_parallel_residual": true,
  "vocab_size": 50688
}
```


```
====================
9820 yes
4374 Yes
24239 YES
====================
4754 Ġyes
26812 ĠBayesian
33275 ĠHayes
43688 Yesterday
4374 Yes
6279 ĠYes
39503 ĠEyes
22487 ĠYES
24239 YES
11066 Ġyesterday
2927 Ġeyes
44011 Ġpolyester
36348 Ġdyes
9820 yes
```