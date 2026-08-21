from tinygrad import Tensor, nn

class TinyTransformerBlock:
    def __init__(self, d_model, num_heads, d_ff):
        # Pre-LN architecture: LayerNorm before attention and before MLP
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        
        # Q, K, V projections for multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        
        # MLP with two linear layers and GELU activation
        self.fc1 = nn.Linear(d_model, d_ff, bias=False)
        self.fc2 = nn.Linear(d_ff, d_model, bias=False)
        
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_head = d_model // num_heads
    
    def __call__(self, x):
        # x: (B, T, d_model)
        B, T, _ = x.shape
        
        # Pre-LN: LayerNorm before attention
        normed = self.ln1(x)
        
        # Multi-head attention
        Q = self.q_proj(normed)  # (B, T, d_model)
        K = self.k_proj(normed)  # (B, T, d_model)
        V = self.v_proj(normed)  # (B, T, d_model)
        
        # Reshape to (B, num_heads, T, d_head)
        Q = Q.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        K = K.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        V = V.reshape(B, T, self.num_heads, self.d_head).transpose(1, 2)
        
        # Scaled dot-product attention with causal masking
        attn_out = Tensor.scaled_dot_product_attention(Q, K, V, is_causal=True)
        
        # Reshape back to (B, T, d_model)
        attn_out = attn_out.transpose(1, 2).reshape(B, T, self.d_model)
        
        # Out projection
        attn_out = self.out_proj(attn_out)
        
        # Residual connection
        x = x + attn_out
        
        # Pre-LN: LayerNorm before MLP
        normed = self.ln2(x)
        
        # MLP with GELU activation
        mlp_out = self.fc1(normed)
        mlp_out = mlp_out.gelu()  # GELU activation
        mlp_out = self.fc2(mlp_out)
        
        # Residual connection
        x = x + mlp_out
        
        return x


class TinyGPT:
    def __init__(self, vocab_size, d_model, num_heads, num_layers, max_seq_len):
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len
        
        # Token and position embeddings
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        
        # Stack of transformer blocks
        d_ff = 4 * d_model
        self.blocks = [
            TinyTransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ]
        
        # Final LayerNorm
        self.ln_f = nn.LayerNorm(d_model)
        
        # Output projection (no bias)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
    
    def __call__(self, idx):
        # idx: (B, T) tensor of token ids
        B, T = idx.shape
        
        # Token embeddings
        x = self.token_emb(idx)  # (B, T, d_model)
        
        # Positional embeddings for positions 0 through T-1
        positions = Tensor.arange(T)  # (T,)
        pos_emb = self.pos_emb(positions)  # (T, d_model)
        
        # Add positional embeddings (broadcast over batch dimension)
        x = x + pos_emb.reshape(1, T, self.d_model)
        
        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x)
        
        # Final LayerNorm
        x = self.ln_f(x)
        
        # Project to vocabulary logits
        logits = self.head(x)  # (B, T, vocab_size)
        
        return logits