import numpy as np

def hybrid_vit_multitask(image, patch_conv_w, patch_conv_b, pos_emb,
                         attn_Wq, attn_Wk, attn_Wv, attn_Wo,
                         num_heads, task_heads):
    """
    Hybrid Vision Transformer for multi-task driving perception.

    Args:
        image: numpy array (H, W, C)
        patch_conv_w: numpy array (D, kH, kW, C)
        patch_conv_b: numpy array (D,)
        pos_emb: numpy array (N, D)
        attn_Wq: numpy array (D, D)
        attn_Wk: numpy array (D, D)
        attn_Wv: numpy array (D, D)
        attn_Wo: numpy array (D, D)
        num_heads: int
        task_heads: dict {name: (W, b)}

    Returns:
        dict {name: numpy array (N, out_dim)}
    """
    H, W, C = image.shape
    D, kH, kW, _ = patch_conv_w.shape

    # Stage 1: Convolutional Patch Embedding
    N_H = H // kH
    N_W = W // kW
    N = N_H * N_W

    # Extract non-overlapping patches (stride = kernel size)
    patches = image[:N_H * kH, :N_W * kW, :].reshape(N_H, kH, N_W, kW, C)
    patches = patches.transpose(0, 2, 1, 3, 4).reshape(N, kH, kW, C)

    # Project patches using learned convolutional filters + bias
    patch_emb = np.einsum('nhwc, dhwc -> nd', patches, patch_conv_w) + patch_conv_b

    # Stage 2: Positional Embedding
    x = patch_emb + pos_emb

    # Stage 3: Multi-Head Self-Attention with Residual Connection
    d_k = D // num_heads

    # Linear projections for Queries, Keys, and Values
    Q = np.matmul(x, attn_Wq)  # (N, D)
    K = np.matmul(x, attn_Wk)  # (N, D)
    V = np.matmul(x, attn_Wv)  # (N, D)

    # Split into multiple attention heads
    Q_h = Q.reshape(N, num_heads, d_k).transpose(1, 0, 2)  # (num_heads, N, d_k)
    K_h = K.reshape(N, num_heads, d_k).transpose(1, 0, 2)  # (num_heads, N, d_k)
    V_h = V.reshape(N, num_heads, d_k).transpose(1, 0, 2)  # (num_heads, N, d_k)

    # Scaled dot-product attention per head
    scale = np.sqrt(d_k)
    scores = np.matmul(Q_h, K_h.transpose(0, 2, 1)) / scale  # (num_heads, N, N)

    # Numerically stable softmax across patch dimension
    scores_max = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - scores_max)
    attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (num_heads, N, N)

    # Concatenate head outputs
    context = np.matmul(attn_weights, V_h)  # (num_heads, N, d_k)
    context_concat = context.transpose(1, 0, 2).reshape(N, D)  # (N, D)

    # Output projection and residual addition from pre-attention tokens
    attn_out = np.matmul(context_concat, attn_Wo)  # (N, D)
    x_out = x + attn_out  # (N, D)

    # Stage 4: Task-Specific Prediction Heads
    predictions = {}
    for task_name, (W_head, b_head) in task_heads.items():
        predictions[task_name] = np.matmul(x_out, W_head) + b_head

    return predictions