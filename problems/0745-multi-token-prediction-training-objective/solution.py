import numpy as np

def mtp_loss(main_logits, main_targets, mtp_logits, mtp_targets, mtp_weight):
    """
    Compute the combined LM + depth-1 Multi-Token Prediction loss.

    Args:
        main_logits: array-like (N, V)
        main_targets: array-like (N,) integer token ids
        mtp_logits: array-like (N-1, V)
        mtp_targets: array-like (N-1,) integer token ids
        mtp_weight: float

    Returns:
        float: total loss rounded to 6 decimals
    """
    main_logits = np.asarray(main_logits, dtype=np.float64)
    main_targets = np.asarray(main_targets, dtype=np.int64)
    mtp_logits = np.asarray(mtp_logits, dtype=np.float64)
    mtp_targets = np.asarray(mtp_targets, dtype=np.int64)

    def cross_entropy(logits, targets):
        if logits.size == 0:
            return 0.0
        # Numerically stable log-softmax
        max_logits = np.max(logits, axis=1, keepdims=True)
        shifted = logits - max_logits
        log_sum_exp = np.log(np.sum(np.exp(shifted), axis=1, keepdims=True))
        log_probs = shifted - log_sum_exp

        n = logits.shape[0]
        target_log_probs = log_probs[np.arange(n), targets]
        return -np.mean(target_log_probs)

    lm_loss = cross_entropy(main_logits, main_targets)
    mtp_loss_val = cross_entropy(mtp_logits, mtp_targets)

    total = lm_loss + mtp_weight * mtp_loss_val
    return round(float(total), 6)