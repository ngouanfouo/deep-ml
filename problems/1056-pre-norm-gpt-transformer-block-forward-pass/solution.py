import numpy as np

def pre_norm_transformer_block(x, params, num_heads):
    """
    Pre-norm Transformer block forward pass.

    Args:
        x: numpy array of shape (batch, seq_len, emb_dim)
        params: dict with keys 'ln1_gamma','ln1_beta','ln2_gamma','ln2_beta',
                'W_q','W_k','W_v','W_o','W_ff1','b_ff1','W_ff2','b_ff2'
        num_heads: int, number of attention heads (emb_dim must be divisible by num_heads)

    Returns:
        numpy array of shape (batch, seq_len, emb_dim)
    """
    x = np.array(x, dtype=float)
    batch, seq_len, emb_dim = x.shape
    d_head = emb_dim // num_heads

    # Helper function for LayerNorm over the last dimension
    def layer_norm(tensor, gamma, beta, eps=1e-5):
        mean = np.mean(tensor, axis=-1, keepdims=True)
        var = np.var(tensor, axis=-1, keepdims=True)
        normed = (tensor - mean) / np.sqrt(var + eps)
        return normed * gamma + beta

    # Helper function for GELU using the tanh approximation
    def gelu(z):
        return 0.5 * z * (1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (z + 0.044715 * (z ** 3))))

    # --- 1. Attention Sub-Block ---
    ln1_out = layer_norm(x, params['ln1_gamma'], params['ln1_beta'], eps=1e-5)

    Q = ln1_out @ params['W_q']
    K = ln1_out @ params['W_k']
    V = ln1_out @ params['W_v']

    Q = Q.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(batch, seq_len, num_heads, d_head).transpose(0, 2, 1, 3)

    scores = (Q @ K.swapaxes(-1, -2)) / np.sqrt(d_head)

    # Use np.where instead of multiplication to avoid 0 * -inf = NaN
    causal_mask = np.triu(np.ones((seq_len, seq_len), dtype=bool), k=1)
    scores = np.where(causal_mask, -np.inf, scores)

    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

    attn_out = attn_weights @ V
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(batch, seq_len, emb_dim)
    h_attn = attn_out @ params['W_o']

    x = x + h_attn

    # --- 2. Feed-Forward Sub-Block ---
    ln2_out = layer_norm(x, params['ln2_gamma'], params['ln2_beta'], eps=1e-5)

    ff1 = ln2_out @ params['W_ff1'] + params['b_ff1']
    ff_act = gelu(ff1)
    h_ffn = ff_act @ params['W_ff2'] + params['b_ff2']

    x = x + h_ffn

    return x