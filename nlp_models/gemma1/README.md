# GEMMA2


```
I ate bread because I was feeling blue. What do you think?


I ate bread because I was feeling blue. What do you think?

I ate bread because I was feeling blue.
```

## Model

```
GemmaForCausalLM(
  (model): GemmaModel(
    (embed_tokens): Embedding(256000, 3072, padding_idx=0)
    (layers): ModuleList(
      (0-27): 28 x GemmaDecoderLayer(
        (self_attn): GemmaSdpaAttention(
          (q_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (k_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (v_proj): Linear(in_features=3072, out_features=4096, bias=False)
          (o_proj): Linear(in_features=4096, out_features=3072, bias=False)
          (rotary_emb): GemmaRotaryEmbedding()
        )
        (mlp): GemmaMLP(
          (gate_proj): Linear(in_features=3072, out_features=24576, bias=False)
          (up_proj): Linear(in_features=3072, out_features=24576, bias=False)
          (down_proj): Linear(in_features=24576, out_features=3072, bias=False)
          (act_fn): PytorchGELUTanh()
        )
        (input_layernorm): GemmaRMSNorm((3072,), eps=1e-06)
        (post_attention_layernorm): GemmaRMSNorm((3072,), eps=1e-06)
      )
    )
    (norm): GemmaRMSNorm((3072,), eps=1e-06)
  )
  (lm_head): Linear(in_features=3072, out_features=256000, bias=False)
)

```

```
GemmaConfig {
  "_name_or_path": "google/gemma-7b",
  "architectures": [
    "GemmaForCausalLM"
  ],
  "attention_bias": false,
  "attention_dropout": 0.0,
  "bos_token_id": 2,
  "eos_token_id": 1,
  "head_dim": 256,
  "hidden_act": "gelu",
  "hidden_activation": "gelu_pytorch_tanh",
  "hidden_size": 3072,
  "initializer_range": 0.02,
  "intermediate_size": 24576,
  "max_position_embeddings": 8192,
  "model_type": "gemma",
  "num_attention_heads": 16,
  "num_hidden_layers": 28,
  "num_key_value_heads": 16,
  "pad_token_id": 0,
  "rms_norm_eps": 1e-06,
  "rope_scaling": null,
  "rope_theta": 10000.0,
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
172060 ▁employes
173003 ▁egyes
200635 Hayes
123951 Yess
175357 YesNo
200076 diyesi
4628 ▁eyes
227798 ▁Egyes
13351 ▁yesterday
3276 yes
50873 ▁Reyes
154368 ▁menyes
218177 byes
65412 Yesterday
168681 ▁Yesus
84955 ▁Bayesian
21869 ▁YES
229064 bayes
214526 yesha
219433 ▁Noyes
80624 ▁dyes
124047 ▁Bayes
207027 Bayesian
7778 ▁yes
206236 ▁Keyes
197593 ▁yester
53071 Eyes
117162 ▁reyes
49846 ▁Hayes
37682 ▁polyester
124066 yesterday
33904 ▁Eyes
211223 ▁menyesuaikan
202425 ▁sayesinde
3553 Yes
54558 ▁Polyester
142981 ▁eyesight
17480 YES
54924 eyes
76725 ▁leyes
98768 yesi
6287 ▁Yes
172913 Polyester
113011 ▁eyeshadow
203066 Bayes
89792 ▁Yesterday
140879 ▁EYES
```