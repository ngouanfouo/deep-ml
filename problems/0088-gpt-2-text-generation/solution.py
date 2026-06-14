import numpy as np

def gen_text(prompt: str, n_tokens_to_generate: int = 40):
    # Load encoder, hyperparameters, and parameters
    encoder, hparams, params = load_encoder_hparams_and_params()
    
    # Encode the prompt
    input_tokens = encoder.encode(prompt)
    
    # Store only the generated tokens (excluding the prompt for output)
    generated_tokens = []
    current_tokens = input_tokens.copy()
    
    # Generate tokens one by one
    for _ in range(n_tokens_to_generate):
        # Get current sequence (up to model's context length)
        seq_tokens = current_tokens[-hparams["n_ctx"]:]
        seq_len = len(seq_tokens)
        
        # Get embeddings
        token_embeds = params["wte"][seq_tokens]
        pos_indices = np.arange(seq_len)
        pos_embeds = params["wpe"][pos_indices]
        
        # Add token and positional embeddings
        x = token_embeds + pos_embeds
        
        # Pass through transformer blocks
        for block in params["blocks"]:
            # Layer norm 1
            ln1_x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
            ln1_x = ln1_x * block["ln_1"]["g"] + block["ln_1"]["b"]
            
            # Multi-head attention
            n_head = hparams["n_head"]
            d_model = x.shape[-1]
            d_head = d_model // n_head
            
            qkv = ln1_x @ block["attn"]["c_attn"]["w"] + block["attn"]["c_attn"]["b"]
            q, k, v = np.split(qkv, 3, axis=-1)
            
            q = q.reshape(seq_len, n_head, d_head).transpose(1, 0, 2)
            k = k.reshape(seq_len, n_head, d_head).transpose(1, 0, 2)
            v = v.reshape(seq_len, n_head, d_head).transpose(1, 0, 2)
            
            scores = q @ k.transpose(0, 2, 1) / np.sqrt(d_head)
            mask = np.triu(np.ones((seq_len, seq_len)), k=1)
            scores = scores - 1e9 * mask
            
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
            
            attn_out = attn_weights @ v
            attn_out = attn_out.transpose(1, 0, 2).reshape(seq_len, d_model)
            attn_out = attn_out @ block["attn"]["c_proj"]["w"] + block["attn"]["c_proj"]["b"]
            
            x = x + attn_out
            
            # Layer norm 2
            ln2_x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
            ln2_x = ln2_x * block["ln_2"]["g"] + block["ln_2"]["b"]
            
            # Feed-forward network
            mlp_hidden = ln2_x @ block["mlp"]["c_fc"]["w"] + block["mlp"]["c_fc"]["b"]
            mlp_hidden = np.maximum(0, mlp_hidden)
            mlp_out = mlp_hidden @ block["mlp"]["c_proj"]["w"] + block["mlp"]["c_proj"]["b"]
            
            x = x + mlp_out
        
        # Final layer norm
        x = (x - np.mean(x, axis=-1, keepdims=True)) / (np.std(x, axis=-1, keepdims=True) + 1e-5)
        x = x * params["ln_f"]["g"] + params["ln_f"]["b"]
        
        # Get logits for last token
        logits = x[-1] @ params["wte"].T
        next_token = np.argmax(logits)
        
        # Append generated token
        generated_tokens.append(next_token)
        current_tokens.append(next_token)
    
    # Decode only the generated tokens
    generated_text = encoder.decode(generated_tokens)
    return generated_text.strip()  # Strip any extra whitespace

def load_encoder_hparams_and_params(model_size: str = "124M", models_dir: str = "models"):
    class DummyBPE:
        def __init__(self):
            self.encoder_dict = {"hello": 1, "world": 2, "<UNK>": 0}

        def encode(self, text: str):
            tokens = text.strip().split()
            return [self.encoder_dict.get(token, self.encoder_dict["<UNK>"]) for token in tokens]

        def decode(self, token_ids: list):
            reversed_dict = {v: k for k, v in self.encoder_dict.items()}
            return " ".join([reversed_dict.get(tok_id, "<UNK>") for tok_id in token_ids])

    hparams = {
        "n_ctx": 1024,
        "n_head": 2
    }

    params = {
        "wte": np.random.rand(3, 10),
        "wpe": np.random.rand(1024, 10),
        "blocks": [{
            "mlp": {
                "c_fc": {"w": np.random.rand(10, 20), "b": np.random.rand(20)},
                "c_proj": {"w": np.random.rand(20, 10), "b": np.random.rand(10)}
            },
            "attn": {
                "c_attn": {"w": np.random.rand(10, 30), "b": np.random.rand(30)},
                "c_proj": {"w": np.random.rand(10, 10), "b": np.random.rand(10)}
            },
            "ln_1": {"g": np.ones(10), "b": np.zeros(10)},
            "ln_2": {"g": np.ones(10), "b": np.zeros(10)},
        }],
        "ln_f": {
            "g": np.ones(10),
            "b": np.zeros(10),
        }
    }

    encoder = DummyBPE()
    return encoder, hparams, params