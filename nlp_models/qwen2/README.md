# Qwen2


```
I ate bread because I was feeling blue. What do you think?


 I felt better because I had a snack. I ate the bread to make me feel better. I ate it to get something

```

## Model

```
Qwen2ForCausalLM(
  (model): Qwen2Model(
    (embed_tokens): Embedding(152064, 3584)
    (layers): ModuleList(
      (0-27): 28 x Qwen2DecoderLayer(
        (self_attn): Qwen2SdpaAttention(
          (q_proj): Linear(in_features=3584, out_features=3584, bias=True)
          (k_proj): Linear(in_features=3584, out_features=512, bias=True)
          (v_proj): Linear(in_features=3584, out_features=512, bias=True)
          (o_proj): Linear(in_features=3584, out_features=3584, bias=False)
          (rotary_emb): Qwen2RotaryEmbedding()
        )
        (mlp): Qwen2MLP(
          (gate_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (up_proj): Linear(in_features=3584, out_features=18944, bias=False)
          (down_proj): Linear(in_features=18944, out_features=3584, bias=False)
          (act_fn): SiLU()
        )
        (input_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Qwen2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Qwen2RMSNorm((3584,), eps=1e-06)
    (rotary_emb): Qwen2RotaryEmbedding()
  )
  (lm_head): Linear(in_features=3584, out_features=152064, bias=False)
)


```

```
Qwen2Config {
  "_name_or_path": "Qwen/Qwen2-7B-Instruct",
  "architectures": [
    "Qwen2ForCausalLM"
  ],
  "attention_dropout": 0.0,
  "bos_token_id": 151643,
  "eos_token_id": 151645,
  "hidden_act": "silu",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 18944,
  "max_position_embeddings": 32768,
  "max_window_layers": 28,
  "model_type": "qwen2",
  "num_attention_heads": 28,
  "num_hidden_layers": 28,
  "num_key_value_heads": 4,
  "rms_norm_eps": 1e-06,
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
9834 Ġyes
19566 :YES
85408 âĢľYes
76894 .YesNo
40798 .Yes
76159 ĠReyes
52422 ĠHayes
6414 Ġeyes
57231 eyes
9454 Yes
94834 =YES
41996 ĠEyes
74849 ĠPolyester
7414 ĠYes
55070 Ġpolyester
84402 .YES
59744 _yes
98134 ĠBayesian
60033 ĠYesterday
50277 Yesterday
14004 YES
57741 "Yes
9693 yes
97071 ,Yes
14080 ĠYES
13671 Ġyesterday
76730 _YES
59565 =yes

```