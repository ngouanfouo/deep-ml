import numpy as np

def bc_transformer_loss(obs: np.ndarray, targets: np.ndarray, weights: dict) -> float:
    """
    Compute the behavioral cloning loss of a single-block transformer policy
    predicting discrete latent actions from a sequence of observation embeddings.

    Args:
        obs: Array of shape (T, d_model) containing observation embeddings.
        targets: Integer array of shape (T,) with ground-truth latent action indices.
        weights: Dict with keys 'W_q', 'W_k', 'W_v', 'W_o', 'W1', 'b1', 'W2', 'b2',
                 'W_head', 'b_head'.

    Returns:
        The mean token-level cross-entropy behavioral cloning loss as a float.
    """
    T, d_model = obs.shape

    # 1. Linear projections for Queries, Keys, Values
    Q = obs @ weights['W_q']  # (T, d_model)
    K = obs @ weights['W_k']  # (T, d_model)
    V = obs @ weights['W_v']  # (T, d_model)

    # 2. Scaled dot-product self-attention scores
    scale = np.sqrt(d_model)
    scores = (Q @ K.T) / scale  # (T, T)

    # Apply causal mask (tokens can only attend to current and previous timesteps)
    causal_mask = np.tril(np.ones((T, T)))
    scores_masked = np.where(causal_mask == 1, scores, -1e9)

    # Numerically stable softmax along sequence dimension
    scores_max = np.max(scores_masked, axis=-1, keepdims=True)
    exp_scores = np.exp(scores_masked - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (T, T)

    # Attended context and output projection
    context = attn_weights @ V  # (T, d_model)
    attn_out = context @ weights['W_o']  # (T, d_model)

    # First Residual Connection
    x1 = obs + attn_out  # (T, d_model)

    # 3. Position-wise MLP (Linear + ReLU + Linear)
    mlp_h = np.maximum(0, x1 @ weights['W1'] + weights['b1'])  # (T, d_ff)
    mlp_out = mlp_h @ weights['W2'] + weights['b2']            # (T, d_model)

    # Second Residual Connection
    x2 = x1 + mlp_out  # (T, d_model)

    # 4. Policy Head
    logits = x2 @ weights['W_head'] + weights['b_head']  # (T, K)

    # 5. Token-level Cross-Entropy Loss
    target_logits = logits[np.arange(T), targets]
    max_logits = np.max(logits, axis=-1, keepdims=True)
    
    # log_sum_exp trick for numerical stability
    log_sum_exp = np.squeeze(max_logits, axis=-1) + np.log(
        np.sum(np.exp(logits - max_logits), axis=-1)
    )
    
    ce_loss = log_sum_exp - target_logits
    mean_loss = np.mean(ce_loss)

    return float(mean_loss)