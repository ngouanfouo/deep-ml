import numpy as np


def _softmax_rows(scores):
    """Numerically stable row-wise softmax."""
    m = np.max(scores, axis=1, keepdims=True)
    e = np.exp(scores - m)
    return e / np.sum(e, axis=1, keepdims=True)


def attn_only_forward(tokens, W):
    """Return (final residual (T, d), writes dict with "embed" and ("head", layer, head) keys)."""
    W_E = np.asarray(W["W_E"], dtype=float)
    W_pos = np.asarray(W["W_pos"], dtype=float)

    tokens = list(tokens)
    T = len(tokens)

    # Initial residual: token embedding + positional embedding.
    x = W_E[tokens] + W_pos[:T]
    x = np.array(x, dtype=float, copy=True)

    writes = {"embed": x.copy()}

    # Causal mask: position i may attend to j <= i.
    mask = np.triu(np.ones((T, T), dtype=bool), 1)

    layers = W.get("layers", [])
    for layer_idx, heads in enumerate(layers):
        layer_input = x  # fixed while computing all heads of this layer
        head_outputs = []

        for head_idx, head in enumerate(heads):
            W_Q = np.asarray(head["W_Q"], dtype=float)
            W_K = np.asarray(head["W_K"], dtype=float)
            W_V = np.asarray(head["W_V"], dtype=float)
            W_O = np.asarray(head["W_O"], dtype=float)

            d_h = W_Q.shape[1]

            Q = layer_input @ W_Q
            K = layer_input @ W_K
            V = layer_input @ W_V

            scores = (Q @ K.T) / np.sqrt(d_h)

            # Apply causal mask with -inf; every row keeps at least its diagonal.
            scores = np.where(mask, -np.inf, scores)

            A = _softmax_rows(scores)
            out = (A @ V) @ W_O  # (T, d)

            writes[("head", layer_idx, head_idx)] = out.copy()
            head_outputs.append(out)

        # Add the sum of heads only after all heads were computed.
        if head_outputs:
            x = layer_input + np.sum(head_outputs, axis=0)

    return x, writes


def direct_logit_attribution(writes, W_U, tok_a, tok_b, pos=-1):
    """Per-component contribution to the logit difference u_a - u_b at position pos."""
    W_U = np.asarray(W_U, dtype=float)
    direction = W_U[:, tok_a] - W_U[:, tok_b]  # (d,)

    attribution = {}
    for key, w in writes.items():
        w = np.asarray(w, dtype=float)
        contribution = float(direction @ w[pos])
        attribution[key] = float(round(contribution, 6))

    return attribution