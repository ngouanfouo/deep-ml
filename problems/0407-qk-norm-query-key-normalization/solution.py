import torch
import torch.nn.functional as F


def qk_norm_attention(Q: torch.Tensor,
                      K: torch.Tensor,
                      V: torch.Tensor,
                      temperature: float = 1.0) -> tuple:
    """
    Apply QK-Norm attention: L2-normalize queries and keys before computing
    scaled dot-product attention.

    Args:
        Q: Query matrix, shape (seq_len_q, d_k)
        K: Key matrix, shape (seq_len_k, d_k)
        V: Value matrix, shape (seq_len_k, d_v)
        temperature: Temperature scaling parameter (default: 1.0)

    Returns:
        Tuple of (attention_output, attention_weights)
    """
    eps = 1e-8

    # 1) L2-normalize each row of Q and K independently.
    #    clamp(min=eps) keeps zero-norm vectors from dividing by zero;
    #    zero vectors stay zero because the numerator is zero.
    q_norm = torch.norm(Q, p=2, dim=-1, keepdim=True).clamp(min=eps)
    k_norm = torch.norm(K, p=2, dim=-1, keepdim=True).clamp(min=eps)
    Q_hat = Q / q_norm
    K_hat = K / k_norm

    # 2) Attention scores from unit vectors, scaled by temperature.
    #    Rows of Q_hat and K_hat have norm 1, so each score lies in [-1, 1].
    scores = (Q_hat @ K_hat.transpose(-2, -1)) / temperature

    # 3) Numerically stable softmax (subtract row max before exp).
    scores = scores - scores.max(dim=-1, keepdim=True).values
    exp_scores = torch.exp(scores)
    attention_weights = exp_scores / exp_scores.sum(dim=-1, keepdim=True)

    # 4) Weighted sum of values.
    attention_output = attention_weights @ V

    return attention_output, attention_weights