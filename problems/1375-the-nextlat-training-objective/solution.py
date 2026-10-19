import numpy as np

def nextlat_loss(H, E_tok, targets, W_out, b_out, Wh, Wx, d,
                 lam_h, lam_kl, beta=1.0):
    H = np.asarray(H, dtype=np.float64)
    E_tok = np.asarray(E_tok, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.int64)
    W_out = np.asarray(W_out, dtype=np.float64)
    b_out = np.asarray(b_out, dtype=np.float64)
    Wh = np.asarray(Wh, dtype=np.float64)
    Wx = np.asarray(Wx, dtype=np.float64)

    T, D = H.shape

    # ---- next-token cross-entropy ----
    logits = H @ W_out + b_out                      # (T, V)
    m = np.max(logits, axis=-1, keepdims=True)
    shifted = logits - m
    log_sum_exp = np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))
    log_probs = shifted - log_sum_exp
    nll = -log_probs[np.arange(T), targets]
    next_token = float(np.mean(nll))

    # ---- rollout terms ----
    starts = [t for t in range(T - d)]              # t + d <= T - 1

    if len(starts) == 0:
        next_h = 0.0
        kl = 0.0
    else:
        def log_softmax(x):
            mx = np.max(x)
            s = x - mx
            return s - np.log(np.sum(np.exp(s)))

        sl1_vals = []
        kl_vals = []

        for t in starts:
            h_hat = H[t].copy()
            for i in range(1, d + 1):
                h_hat = np.tanh(h_hat @ Wh + E_tok[t + i] @ Wx)
                target_h = H[t + i]

                # Smooth L1 (PyTorch convention)
                diff = h_hat - target_h
                abs_diff = np.abs(diff)
                sl1 = np.where(
                    abs_diff < beta,
                    0.5 * diff ** 2 / beta,
                    abs_diff - 0.5 * beta,
                )
                sl1_vals.append(np.mean(sl1))

                # KL(softmax(logits(H[t+i])) || softmax(logits(h_hat)))
                logit_p = target_h @ W_out + b_out
                logit_q = h_hat @ W_out + b_out
                lp = log_softmax(logit_p)
                lq = log_softmax(logit_q)
                p = np.exp(lp)
                kl_vals.append(np.sum(p * (lp - lq)))

        next_h = float(np.mean(sl1_vals))
        kl = float(np.mean(kl_vals))

    total = next_token + lam_h * next_h + lam_kl * kl

    return {
        'next_token': next_token,
        'next_h': next_h,
        'kl': kl,
        'total': float(total),
    }