# Deon's LLM experiments

```python
General
    prompts = [
        f"Determine if the following sentence is deontic by considering the context and the semantic meaning.\n"
        "Answer with 1 if it's a deontic sentence, 0 if not.you must answer with 1 or 0.\n\n"
        f"Sentence: Context: {ctx}\nInput: {inp}\n\nAnswer: "
        for ctx, inp in zip(contexts, inputs)
    ]

    prompts = [
        f"Determine if the following sentence is deontic by considering the semantic meaning.\n"
        "You must answer with ONLY 1 or 0. Answer with 1 if it's a deontic sentence, 0 if not.\n\n"
        f"Sentence: {item['input']}\n\nAnswer: "
        for item in batch_requests
    ]

Explicit
  prompts = [
    f"Determine whether the following sentence is an obligation based on its context and semantic meaning.\n"
    "Answer with 1 if it's a obligation sentence, 0 if not. you must answer with 1 or 0.\n\n"
    f"Sentence: Context: {ctx}\nInput: {inp}\n\nAnswer: "
    for ctx, inp in zip(contexts, inputs)
  ]

Strict
  prompts = [
    f"Determine whether this sentence mandates compliance in all cases by considering the context and the semantic meaning.\n"
    "Answer with 1 if it conveys a mandatory requirement, 0 if not. You must answer with 1 or 0.\n\n"
    f"Sentence: Context: {ctx}\nInput: {inp}\n\nAnswer: "
    for ctx, inp in zip(contexts, inputs)
  ]
















```