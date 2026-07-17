import torch

@torch.no_grad()
def muonclip_qk_clip(W_q: torch.Tensor, W_k: torch.Tensor, x: torch.Tensor, t: float, alpha: float = 0.5, eps: float = 1e-7):
    """
    PyTorch version of MuonClip qk-clip (no grad). 
    Returns (W_q_new, W_k_new, clipped, max_post).
    """
    # 1. Compute queries and keys projection matrices
    q = torch.matmul(x, W_q)
    k = torch.matmul(x, W_k)
    
    # 2. Compute scaled pre-softmax dot-product attention scores
    d_head = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_head ** 0.5)
    
    # Find maximum pre-clip activation value
    max_score = torch.max(scores).item()
    
    clipped = False
    if max_score > t:
        clipped = True
        # Compute weight clipping scale parameter 
        eta = t / (max_score + eps)
        
        # Apply decoupled geometric weight scaling factor
        W_q_new = W_q * (eta ** alpha)
        W_k_new = W_k * (eta ** (1 - alpha))
        max_post = t
    else:
        W_q_new = W_q.clone()
        W_k_new = W_k.clone()
        max_post = max_score
        
    # Round all dynamic tensor parameters and metrics for stable evaluation tracking
    return (
        torch.round(W_q_new, decimals=4), 
        torch.round(W_k_new, decimals=4), 
        clipped, 
        round(max_post, 4)
    )