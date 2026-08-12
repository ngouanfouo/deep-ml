import torch
from typing import List, Tuple

def fit_bradley_terry(comparisons: List[Tuple[int, int]], n_items: int,
                      learning_rate: float = 0.5, n_iterations: int = 100) -> torch.Tensor:
    """
    Fit Bradley-Terry model parameters using maximum likelihood estimation.

    Args:
        comparisons: List of (winner_idx, loser_idx) tuples
        n_items: Total number of items to rank
        learning_rate: Step size for gradient ascent
        n_iterations: Number of optimization iterations

    Returns:
        torch.Tensor: Estimated strength parameters of shape (n_items,)
    """
    # Initialize parameters to 0
    beta = torch.zeros(n_items, dtype=torch.float32)
    
    # Convert comparisons to tensors for efficient processing
    winners = torch.tensor([w for w, _ in comparisons], dtype=torch.long)
    losers = torch.tensor([l for _, l in comparisons], dtype=torch.long)
    
    for iteration in range(n_iterations):
        # Get parameters for all comparisons
        beta_w = beta[winners]  # strength of winners
        beta_l = beta[losers]   # strength of losers
        
        # Compute difference: beta_l - beta_w
        # This represents the log-odds of loser beating winner
        diff = beta_l - beta_w
        
        # Numerically stable sigmoid
        # sigmoid(diff) = 1 / (1 + exp(-diff))
        mask = diff >= 0
        sigmoid_vals = torch.where(
            mask,
            1.0 / (1.0 + torch.exp(-diff)),
            torch.exp(diff) / (1.0 + torch.exp(diff))
        )
        
        # Initialize gradients
        grad = torch.zeros(n_items, dtype=torch.float32)
        
        # For winner w and loser l:
        # grad[w] += sigmoid(beta_l - beta_w)  # positive gradient for winner
        # grad[l] += -sigmoid(beta_l - beta_w) # negative gradient for loser
        grad.scatter_add_(0, winners, sigmoid_vals)
        grad.scatter_add_(0, losers, -sigmoid_vals)
        
        # Update parameters using gradient ascent
        beta = beta + learning_rate * grad
        
        # Center parameters (subtract mean) for identifiability
        beta = beta - beta.mean()
    
    return beta