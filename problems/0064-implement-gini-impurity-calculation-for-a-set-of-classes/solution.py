import torch

def gini_impurity(y: torch.Tensor) -> float:
    """
    Calculate Gini Impurity for a tensor of class labels.

    :param y: 1D Tensor of class labels (integer type)
    :return: Gini Impurity rounded to three decimal places
    """
    # Count occurrences of each class
    counts = torch.bincount(y)
    total = counts.sum().float()
    # Compute probabilities
    probs = counts / total
    # Gini impurity = 1 - sum(p_i^2)
    gini = 1.0 - torch.sum(probs ** 2)
    return round(gini.item(), 3)