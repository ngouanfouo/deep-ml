import torch

def xgboost_objective(gradients: torch.Tensor, hessians: torch.Tensor,
                      left_indices: torch.Tensor, right_indices: torch.Tensor,
                      lambda_reg: float = 1.0, gamma: float = 0.0) -> dict:
    """
    Calculate XGBoost objective function components for a potential split.
    
    Args:
        gradients: First-order gradients for each sample (torch.Tensor)
        hessians: Second-order hessians for each sample (torch.Tensor)
        left_indices: Indices of samples going to left child (torch.Tensor, long)
        right_indices: Indices of samples going to right child (torch.Tensor, long)
        lambda_reg: L2 regularization parameter
        gamma: Tree complexity penalty
        
    Returns:
        Dictionary with 'left_weight', 'right_weight', and 'gain'
    """
    # Convert indices to long type if needed
    if left_indices.dtype != torch.long:
        left_indices = left_indices.long()
    if right_indices.dtype != torch.long:
        right_indices = right_indices.long()
    
    # Compute sums for left child
    G_L = torch.sum(gradients[left_indices])
    H_L = torch.sum(hessians[left_indices])
    
    # Compute sums for right child
    G_R = torch.sum(gradients[right_indices])
    H_R = torch.sum(hessians[right_indices])
    
    # Compute optimal leaf weights
    # Weight = -sum(gradients) / (sum(hessians) + lambda_reg)
    left_weight = -G_L / (H_L + lambda_reg)
    right_weight = -G_R / (H_R + lambda_reg)
    
    # Compute total gradient and hessian for parent
    G_total = G_L + G_R
    H_total = H_L + H_R
    
    # Compute gain
    # Gain = 0.5 * [G_L^2/(H_L+lambda) + G_R^2/(H_R+lambda) - G_total^2/(H_total+lambda)] - gamma
    gain = 0.5 * (G_L**2 / (H_L + lambda_reg) + 
                  G_R**2 / (H_R + lambda_reg) - 
                  G_total**2 / (H_total + lambda_reg)) - gamma
    
    # Round to 4 decimal places
    return {
        'left_weight': round(left_weight.item(), 4),
        'right_weight': round(right_weight.item(), 4),
        'gain': round(gain.item(), 4)
    }