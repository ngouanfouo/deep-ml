import numpy as np

def log_softmax(scores):
    # Subtract max for numerical stability
    max_score = np.max(scores)
    stable_scores = scores - max_score
    
    # Compute log-sum-exp: log(sum(exp(stable_scores)))
    log_sum_exp = np.log(np.sum(np.exp(stable_scores)))
    
    # log-softmax = scores - max - log(sum(exp(scores - max)))
    log_softmax_values = stable_scores - log_sum_exp
    
    return log_softmax_values