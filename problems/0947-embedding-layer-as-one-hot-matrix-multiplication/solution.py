import numpy as np

def embedding_via_one_hot(token_ids, W):
    token_ids = np.asarray(token_ids, dtype=int)
    W = np.asarray(W)

    num_tokens = len(token_ids)
    vocab_size = W.shape[0]

    # Build one-hot matrix H of shape (num_tokens, vocab_size)
    H = np.zeros((num_tokens, vocab_size), dtype=float)
    H[np.arange(num_tokens), token_ids] = 1.0

    # Matrix multiplication selects the corresponding rows of W
    return H @ W