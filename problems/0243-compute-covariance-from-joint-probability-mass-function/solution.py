import torch

def covariance_from_joint_pmf(x_values: list, y_values: list, joint_pmf: torch.Tensor) -> float:
    """
    Compute the covariance of X and Y from their joint PMF.
    
    Args:
        x_values: List of possible values for X
        y_values: List of possible values for Y
        joint_pmf: 2D torch tensor where joint_pmf[i][j] = P(X=x_values[i], Y=y_values[j])
    
    Returns:
        Covariance of X and Y as a float
    """
    # Convert to torch tensors if they're not already
    x = torch.tensor(x_values, dtype=torch.float32)
    y = torch.tensor(y_values, dtype=torch.float32)
    
    # Compute marginal probabilities for X
    # P(X=x_i) = sum over j of joint_pmf[i][j]
    px = torch.sum(joint_pmf, dim=1)
    
    # Compute marginal probabilities for Y
    # P(Y=y_j) = sum over i of joint_pmf[i][j]
    py = torch.sum(joint_pmf, dim=0)
    
    # Compute E[X] = sum_i x_i * P(X=x_i)
    ex = torch.sum(x * px)
    
    # Compute E[Y] = sum_j y_j * P(Y=y_j)
    ey = torch.sum(y * py)
    
    # Compute E[XY] = sum_i sum_j x_i * y_j * P(X=x_i, Y=y_j)
    # Create outer product of x and y
    xy = torch.outer(x, y)
    exy = torch.sum(xy * joint_pmf)
    
    # Compute covariance: Cov(X,Y) = E[XY] - E[X] * E[Y]
    covariance = exy - ex * ey
    
    # Convert to Python float
    return float(covariance)