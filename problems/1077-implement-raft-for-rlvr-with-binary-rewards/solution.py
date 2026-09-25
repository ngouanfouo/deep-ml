import numpy as np

def raft_rlvr_loss(token_logprobs, rewards):
    """
    Compute the RAFT-for-RLVR supervised fine-tuning loss.

    Args:
        token_logprobs: list[list[list[float]]] of shape [P][K][T_pk]
        rewards: list[list[int]] of shape [P][K] with binary values.

    Returns:
        float: mean per-response negative mean log-likelihood over kept responses,
               or 0.0 if no response was kept.
    """
    per_response_nlls = []

    for prompt_logprobs, prompt_rewards in zip(token_logprobs, rewards):
        for response_logprobs, reward in zip(prompt_logprobs, prompt_rewards):
            if reward > 0:
                # Negative mean log-likelihood for this response
                nll = -float(np.mean(response_logprobs))
                per_response_nlls.append(nll)

    if not per_response_nlls:
        return 0.0

    return float(np.mean(per_response_nlls))