import torch
import torch.nn.functional as F

def dpo_loss(log_probs_chosen_policy: list, log_probs_rejected_policy: list,
             log_probs_chosen_ref: list, log_probs_rejected_ref: list,
             beta: float) -> dict:
    """
    Compute the Direct Preference Optimization (DPO) loss using PyTorch.
    
    Args:
        log_probs_chosen_policy: Log-probs of chosen responses under policy
        log_probs_rejected_policy: Log-probs of rejected responses under policy
        log_probs_chosen_ref: Log-probs of chosen responses under reference model
        log_probs_rejected_ref: Log-probs of rejected responses under reference model
        beta: Temperature parameter for KL constraint strength
    
    Returns:
        Dictionary with 'loss' (float), 'chosen_rewards' (list), and 'rejected_rewards' (list)
    """
    # Convert inputs to torch tensors
    pi_chosen = torch.tensor(log_probs_chosen_policy, dtype=torch.float32)
    pi_rejected = torch.tensor(log_probs_rejected_policy, dtype=torch.float32)
    ref_chosen = torch.tensor(log_probs_chosen_ref, dtype=torch.float32)
    ref_rejected = torch.tensor(log_probs_rejected_ref, dtype=torch.float32)
    
    # 1. Compute implicit rewards: r(x, y) = beta * log(pi(y|x) / ref(y|x))
    chosen_rewards = beta * (pi_chosen - ref_chosen)
    rejected_rewards = beta * (pi_rejected - ref_rejected)
    
    # 2. Compute logits for the preference classification loss: beta * (chosen_log_ratio - rejected_log_ratio)
    logits = chosen_rewards - rejected_rewards
    
    # 3. DPO loss is the negative log-sigmoid of the logits
    loss = -F.logsigmoid(logits).mean()
    
    # 4. Format outputs as specified (round to 4 decimal places, convert lists back)
    return {
        'loss': round(float(loss.item()), 4),
        'chosen_rewards': [round(float(r), 4) for r in chosen_rewards.tolist()],
        'rejected_rewards': [round(float(r), 4) for r in rejected_rewards.tolist()]
    }