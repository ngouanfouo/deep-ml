import torch
import torch.nn as nn

class TinyGPT(nn.Module):
    def __init__(self, vocab_size: int, d_model: int, num_heads: int, num_layers: int, max_seq_len: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        
        # Learned positional embeddings
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Stack of transformer blocks (pre-LN)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=num_heads,
                dim_feedforward=4 * d_model,
                activation='gelu',
                batch_first=True,
                norm_first=True,
                dropout=0.0
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection (no bias)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def forward(self, idx):
        # idx: (B, T) tensor of token ids
        B, T = idx.shape
        
        # Generate causal mask
        # Create upper triangular matrix of -inf above the diagonal
        mask = torch.triu(torch.ones(T, T, device=idx.device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        
        # Token embeddings
        x = self.token_emb(idx)  # (B, T, d_model)
        
        # Positional embeddings for positions 0 through T-1
        positions = torch.arange(T, device=idx.device)  # (T,)
        pos_emb = self.pos_emb(positions)  # (T, d_model)
        
        # Add positional embeddings (broadcast over batch dimension)
        x = x + pos_emb.unsqueeze(0)  # (B, T, d_model)
        
        # Pass through transformer blocks with causal mask
        for block in self.blocks:
            x = block(x, src_mask=mask)  # (B, T, d_model)
        
        # Final layer norm
        x = self.ln_f(x)  # (B, T, d_model)
        
        # Project to vocabulary logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits