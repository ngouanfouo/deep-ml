import torch

def flash_attention_forward(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, 
                            block_size: int = 2) -> torch.Tensor:
    """
    Compute attention output using Flash Attention v1 algorithm.
    
    Args:
        Q: Query matrix (seq_len, d_model)
        K: Key matrix (seq_len, d_model)
        V: Value matrix (seq_len, d_model)
        block_size: Size of blocks for tiled computation
        
    Returns:
        Output matrix (seq_len, d_model)
    """
    seq_len, d_model = Q.shape
    device = Q.device
    dtype = Q.dtype
    
    # Scaling factor as used in standard scaled dot-product attention
    scale = 1.0 / (d_model ** 0.5)
    
    # Initialize output and running statistics for online softmax
    # O: accumulated weighted values (seq_len, d_model)
    O = torch.zeros((seq_len, d_model), dtype=dtype, device=device)
    
    # m: running maximum for each query row (-inf initialized)
    m = torch.full((seq_len, 1), float('-inf'), dtype=dtype, device=device)
    
    # l: running sum of exponentials for each query row (denominator)
    l = torch.zeros((seq_len, 1), dtype=dtype, device=device)
    
    # Number of blocks along the sequence length
    num_blocks = (seq_len + block_size - 1) // block_size
    
    # Outer loop over column blocks of K and V
    for j in range(num_blocks):
        start_j = j * block_size
        end_j = min(start_j + block_size, seq_len)
        
        K_block = K[start_j:end_j]  # (block_size, d_model)
        V_block = V[start_j:end_j]  # (block_size, d_model)
        
        # Compute scaled dot-product attention scores for the current block: Q * K_block^T
        # Shape: (seq_len, block_size)
        scores = torch.matmul(Q, K_block.transpose(0, 1)) * scale
        
        # Compute maximum score in the current block for each row
        block_m = torch.max(scores, dim=1, keepdim=True).values  # (seq_len, 1)
        
        # Update running max: m_new = max(m, block_m)
        m_new = torch.maximum(m, block_m)  # (seq_len, 1)
        
        # Compute exponential of adjusted scores for current block
        # exp(scores - m_new)
        exp_scores = torch.exp(scores - m_new)  # (seq_len, block_size)
        
        # Correction factors for previous blocks and new block sum
        # alpha accounts for the shift in the max value
        alpha = torch.exp(m - m_new)  # (seq_len, 1)
        block_l = torch.sum(exp_scores, dim=1, keepdim=True)  # (seq_len, 1)
        
        # Update running denominator l: l_new = alpha * l + block_l
        l_new = alpha * l + block_l  # (seq_len, 1)
        
        # Update running output O:
        # O_new = diag(alpha)^(-1) * ... wait, proper scaling is: O = alpha * O + exp_scores * V_block
        O = alpha * O + torch.matmul(exp_scores, V_block)
        
        # Update tracking stats
        m = m_new
        l = l_new
        
    # Final normalization by the accumulated denominator sum
    O = O / l
    
    return O