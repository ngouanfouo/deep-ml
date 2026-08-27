import torch

def raft_alignment(
    stages_log_probs: list[list[list[list[float]]]],
    stages_rewards: list[list[list[float]]],
) -> list[float]:
    """
    Simulate the RAFT alignment loop over T stages using PyTorch.
    """
    # Convert to PyTorch tensors for easier processing
    # stages_log_probs: [T][num_prompts][K][num_tokens]
    # stages_rewards: [T][num_prompts][K]
    
    stage_losses = []
    
    for t in range(len(stages_log_probs)):
        # Get log probs and rewards for this stage
        stage_log_probs = stages_log_probs[t]  # [num_prompts][K][num_tokens]
        stage_rewards = stages_rewards[t]      # [num_prompts][K]
        
        num_prompts = len(stage_log_probs)
        total_loss = 0.0
        
        for i in range(num_prompts):
            # Find the response with the highest reward for this prompt
            rewards_for_prompt = stage_rewards[i]  # [K]
            best_response_idx = torch.argmax(torch.tensor(rewards_for_prompt)).item()
            
            # Get the log probs of the best response
            best_response_log_probs = stage_log_probs[i][best_response_idx]  # [num_tokens]
            
            # Compute negative log-likelihood (NLL) for this response
            # NLL = -sum(log_probs)
            nll = -torch.sum(torch.tensor(best_response_log_probs))
            
            # Average over tokens
            num_tokens = len(best_response_log_probs)
            avg_nll = nll / num_tokens
            
            total_loss += avg_nll
        
        # Average loss across all prompts for this stage
        stage_avg_loss = total_loss / num_prompts
        
        # Round to 4 decimal places
        stage_losses.append(round(float(stage_avg_loss), 4))
    
    return stage_losses