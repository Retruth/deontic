# QwQ


```
I ate bread because I was feeling blue. What do you think?

<need to add>

```

## Model

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 5120)
    (layers): ModuleList(
      (0-63): 64 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=5120, out_features=5120, bias=True)
          (k_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (v_proj): Linear(in_features=5120, out_features=1024, bias=True)
          (o_proj): Linear(in_features=5120, out_features=5120, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=5120, out_features=27648, bias=False)
          (up_proj): Linear(in_features=5120, out_features=27648, bias=False)
          (down_proj): Linear(in_features=27648, out_features=5120, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
        (post_attention_layernorm): Qwen2RMSNorm((5120,), eps=1e-05)
      )
    )
    (norm): Qwen2RMSNorm((5120,), eps=1e-05)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=5120, out_features=152064, bias=False)
)


```

```
Qwen2Config {
  "_name_or_path": "Qwen/QwQ-32B-Preview",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 5120,
  "initializer_range": 0.02,
  "intermediate_size": 27648,
  "max_position_embeddings": 32768,
  "max_window_layers": 64,
  "model_type": "qwen2",
  "num_attention_heads": 40,
  "num_hidden_layers": 64,
  "num_key_value_heads": 8,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "rope_theta": 1000000.0,
  "sliding_window": null,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "use_sliding_window": false,
  "vocab_size": 152064
}
```


```
====================
9693 yes
9454 Yes
14004 YES
====================
59744 _yes
57741 "Yes
7414 ĠYes
50277 Yesterday
97071 ,Yes
55070 Ġpolyester
9454 Yes
19566 :YES
98134 ĠBayesian
76730 _YES
60033 ĠYesterday
40798 .Yes
13671 Ġyesterday
84402 .YES
85408 âĢľYes
57231 eyes
76159 ĠReyes
6414 Ġeyes
52422 ĠHayes
14080 ĠYES
74849 ĠPolyester
9693 yes
9834 Ġyes
76894 .YesNo
41996 ĠEyes
14004 YES
94834 =YES
59565 =yes
```