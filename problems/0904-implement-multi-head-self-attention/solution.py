import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int):
        super().__init__()
        # Assert that d_model is divisible by num_heads
        assert d_model % num_heads == 0, f"d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        
        # Create bias-free linear projections
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
    def forward(self, x, mask=None):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        
        # Project to queries, keys, values
        Q = self.q_proj(x)  # (B, T, d_model)
        K = self.k_proj(x)  # (B, T, d_model)
        V = self.v_proj(x)  # (B, T, d_model)
        
        # Reshape to (B, num_heads, T, d_head)
        Q = Q.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, T, d_head)
        K = K.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, T, d_head)
        V = V.view(B, T, self.num_heads, self.d_head).transpose(1, 2)  # (B, num_heads, T, d_head)
        
        # Compute scaled dot-product attention scores
        # Q @ K^T / sqrt(d_head)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_head ** 0.5)  # (B, num_heads, T, T)
        
        # Apply mask if provided
        if mask is not None:
            # mask shape: (T, T) - add it to scores before softmax
            scores = scores + mask
        
        # Softmax along the last axis (key dimension)
        attn_weights = F.softmax(scores, dim=-1)  # (B, num_heads, T, T)
        
        # Apply attention weights to values
        out = torch.matmul(attn_weights, V)  # (B, num_heads, T, d_head)
        
        # Reshape back to (B, T, d_model)
        out = out.transpose(1, 2).contiguous().view(B, T, self.d_model)  # (B, T, d_model)
        
        # Project through out_proj
        out = self.out_proj(out)  # (B, T, d_model)
        
        return out