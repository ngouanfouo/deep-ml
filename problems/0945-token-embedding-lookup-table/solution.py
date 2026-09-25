import numpy as np

def token_embedding_lookup(vocab_size: int, embed_dim: int, token_ids: list, seed: int = 0) -> list:
    rng = np.random.default_rng(seed)
    embedding_table = rng.standard_normal((vocab_size, embed_dim))
    return embedding_table[token_ids].tolist()