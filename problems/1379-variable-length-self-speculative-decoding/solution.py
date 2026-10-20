import numpy as np


def self_speculative_decode(H_true, target_tokens, E, W_out, b_out, Wh, Wx, max_draft):
    T = len(target_tokens)
    if T == 0:
        return 0, 0.0

    def tau(h, e):
        return np.tanh(h @ Wh + e @ Wx)

    pos = 0
    num_cycles = 0
    accepted_counts = []

    while pos < T:
        # The verifier's own token is always accepted (not counted).
        h = tau(H_true[pos], E[target_tokens[pos]])

        # Draft up to max_draft further tokens by recursive greedy decoding.
        draft = []
        for i in range(max_draft):
            if pos + 1 + i >= T:
                break
            logits = h @ W_out + b_out
            token = int(np.argmax(logits))  # ties -> smallest index
            draft.append(token)
            h = tau(h, E[token])

        # Accept the longest prefix of the draft matching the verifier.
        n = 0
        while n < len(draft) and draft[n] == target_tokens[pos + 1 + n]:
            n += 1

        accepted_counts.append(n)
        num_cycles += 1
        pos += 1 + n

    mean_accepted = float(np.mean(accepted_counts))
    return num_cycles, mean_accepted