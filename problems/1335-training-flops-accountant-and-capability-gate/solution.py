import math

def capability_gate(n_params, n_tokens, log10_compute_thresholds, eval_scores, eval_limits):
    """
    Account training FLOPs and return a compute / eval capability gate.
    
    Args:
        n_params: number of model parameters (strictly positive)
        n_tokens: number of training tokens (strictly positive)
        log10_compute_thresholds: list of log10 thresholds
        eval_scores: dict of evaluation score names to values
        eval_limits: dict of evaluation limit names to thresholds
    
    Returns:
        dict with keys: 'log10_flops', 'compute_band', 'flagged_evals', 'decision'
    """
    L = math.log10(6.0 * n_params * n_tokens)
    log10_flops = round(L, 4)

    compute_band = sum(1 for t in log10_compute_thresholds if L >= t)

    # Find eval names present in both dicts and where score >= limit
    common = set(eval_scores.keys()) & set(eval_limits.keys())
    flagged_evals = sorted([name for name in common if eval_scores[name] >= eval_limits[name]])

    if flagged_evals or compute_band >= 2:
        decision = "pause"
    elif compute_band == 1 and not flagged_evals:
        decision = "report"
    else:
        decision = "below"

    return {
        "log10_flops": log10_flops,
        "compute_band": compute_band,
        "flagged_evals": flagged_evals,
        "decision": decision
    }