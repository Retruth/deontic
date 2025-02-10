## ExaOne

ExaOne is a model that can be used to generate text. It is a fine-tuned version of the ExaOne model.

```bash 
I ate bread because I was feeling blue. What do you think?

Person 2:
I'm sorry to hear that. Bread can be comforting, but sometimes it's

```

### Structure

```python
ExaoneForCausalLM(
  (transformer): ExaoneModel(
    (wte): Embedding(102400, 4096, padding_idx=0)
    (drop): Dropout(p=0.0, inplace=False)
    (h): ModuleList(
      (0-31): 32 x ExaoneBlock(
        (ln_1): ExaoneRMSNorm()
        (attn): ExaoneAttention(
          (attention): ExaoneSdpaAttention(
            (rotary): ExaoneRotaryEmbedding()
            (k_proj): Linear(in_features=4096, out_features=1024, bias=False)
            (v_proj): Linear(in_features=4096, out_features=1024, bias=False)
            (q_proj): Linear(in_features=4096, out_features=4096, bias=False)
            (out_proj): Linear(in_features=4096, out_features=4096, bias=False)
          )
        )
        (ln_2): ExaoneRMSNorm()
        (mlp): ExaoneGatedMLP(
          (c_fc_0): Linear(in_features=4096, out_features=14336, bias=False)
          (c_fc_1): Linear(in_features=4096, out_features=14336, bias=False)
          (c_proj): Linear(in_features=14336, out_features=4096, bias=False)
          (act): SiLU()
        )
      )
    )
    (ln_f): ExaoneRMSNorm()
    (rotary): ExaoneRotaryEmbedding()
  )
  (lm_head): Linear(in_features=4096, out_features=102400, bias=False)
)
```

### Configuration

```python
ExaoneConfig {
  "_name_or_path": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct",
  "activation_function": "silu",
  "architectures": [
    "ExaoneForCausalLM"
  ],
  "attention_dropout": 0.0,
  "auto_map": {
    "AutoConfig": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct--configuration_exaone.ExaoneConfig",
    "AutoModelForCausalLM": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct--modeling_exaone.ExaoneForCausalLM",
    "AutoModelForSequenceClassification": "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct--modeling_exaone.ExaoneForSequenceClassification"
  },
  "bos_token_id": 1,
  "embed_dropout": 0.0,
  "eos_token_id": 361,
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 14336,
  "layer_norm_epsilon": 1e-05,
  "max_position_embeddings": 4096,
  "model_type": "exaone",
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "num_layers": 32,
  "pad_token_id": 0,
  "rope_scaling": null,
  "rope_theta": 500000.0,
  "tie_word_embeddings": false,
  "torch_dtype": "bfloat16",
  "transformers_version": "4.45.2",
  "use_cache": true,
  "vocab_size": 102400
}
```

----

## Token Examples

```bash 
====================
64173 yes
17050 Yes
====================
8985 Ġyes
5863 Ġeyes
8277 ĠYes
64376 ĠBayes
52847 Ġpolyester
52217 ĠEyes
34687 ĠBayesian
49495 Ġdyes
72405 eyes
38501 ĠYES
74897 Yesterday
18509 Ġyesterday
64173 yes
69035 ĠYesterday
50711 ĠHayes
72984 ĠReyes
17050 Yes
```