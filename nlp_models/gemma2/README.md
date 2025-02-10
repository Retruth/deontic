# GEMMA2


```
I ate bread because I was feeling blue. What do you think?



[User 0001]

I ate bread because I was feeling blue. What do you think?
```

## Model

```
Gemma2ForCausalLM(
  (model): Gemma2Model(
    (embed_tokens): Embedding(256000, 3584, padding_idx=0)
    (layers): ModuleList(
      (0-41): 42 x Gemma2DecoderLayer(
        (self_attn): Gemma2Attention(
          (q_proj): Linear(in_features=3584, out_features=4096, bias=False)
          (k_proj): Linear(in_features=3584, out_features=2048, bias=False)
          (v_proj): Linear(in_features=3584, out_features=2048, bias=False)
          (o_proj): Linear(in_features=4096, out_features=3584, bias=False)
          (rotary_emb): Gemma2RotaryEmbedding()
        )
        (mlp): Gemma2MLP(
          (gate_proj): Linear(in_features=3584, out_features=14336, bias=False)
          (up_proj): Linear(in_features=3584, out_features=14336, bias=False)
          (down_proj): Linear(in_features=14336, out_features=3584, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (pre_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (post_feedforward_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
        (post_attention_layernorm): Gemma2RMSNorm((3584,), eps=1e-06)
      )
    )
    (norm): Gemma2RMSNorm((3584,), eps=1e-06)
  )
  (lm_head): Linear(in_features=3584, out_features=256000, bias=False)
)

```

```
Gemma2Config {
  "_name_or_path": "google/gemma-2-9b",
  "architectures": [
    "Gemma2ForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "attn_logit_softcapping": 50.0,
  "bos_token_id": 2,
  "cache_implementation": "hybrid",
  "eos_token_id": 1,
  "final_logit_softcapping": 30.0,
  "head_dim": 256,
  "hidden_act": "gelu_pytorch_tanh",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 3584,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "max_position_embeddings": 8192,
  "model_type": "gemma2",
  "num_attention_heads": 16,
  "num_hidden_layers": 42,
  "num_key_value_heads": 8,
  "pad_token_id": 0,
  "query_pre_attn_scalar": 256,
  "rms_norm_eps": 1e-06,
  "rope_theta": 10000.0,
  "sliding_window": 4096,
  "sliding_window_size": 4096,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 256000
}
```


```
====================
3276 yes
3553 Yes
17480 YES
====================
227798 ▁Egyes
142981 ▁eyesight
50873 ▁Reyes
202425 ▁sayesinde
117162 ▁reyes
80624 ▁dyes
3553 Yes
89792 ▁Yesterday
13351 ▁yesterday
229064 bayes
219433 ▁Noyes
172060 ▁employes
211223 ▁menyesuaikan
37682 ▁polyester
200076 diyesi
168681 ▁Yesus
154368 ▁menyes
200635 Hayes
206236 ▁Keyes
6287 ▁Yes
33904 ▁Eyes
53071 Eyes
175357 YesNo
207027 Bayesian
65412 Yesterday
140879 ▁EYES
7778 ▁yes
173003 ▁egyes
54558 ▁Polyester
17480 YES
76725 ▁leyes
124066 yesterday
214526 yesha
123951 Yess
84955 ▁Bayesian
21869 ▁YES
218177 byes
54924 eyes
4628 ▁eyes
3276 yes
49846 ▁Hayes
124047 ▁Bayes
172913 Polyester
113011 ▁eyeshadow
203066 Bayes
197593 ▁yester
98768 yesi

```