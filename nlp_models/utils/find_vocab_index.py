def find_vocab_index(tokenizer, token):
    indices = []
    for t in [token.lower(), token.title(), token.upper()]:
        try:
            indices.append(tokenizer.vocab[t])
        except KeyError:
            indices.append(None)
    return indices

def find_inclusion_vocab_index(tokenizer, token):
    indices = []
    search_token = token.lower()
    for vocab_token in tokenizer.vocab:
        if search_token in vocab_token.lower():
            indices.append(tokenizer.vocab[vocab_token])
    return indices


def decode_token_index(tokenizer, index):
    return tokenizer.convert_ids_to_tokens(index)