import torch
import torch.nn.functional as F

def thanksgiving_dish_predictor(preference_scores: torch.Tensor) -> torch.Tensor:
    """
    Predict the probability of choosing each Thanksgiving dish using softmax.
    
    Args:
        preference_scores: Tensor of preference scores for each dish
        
    Returns:
        Tensor of probabilities for each dish
    """
    # Apply softmax along the last dimension to compute probabilities
    return F.softmax(preference_scores, dim=-1)