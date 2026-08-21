import torch
import torch.nn.functional as F

def calculate_oob_score(n_samples: int, bootstrap_indices: list, predictions: torch.Tensor, y_true: torch.Tensor) -> float:
    """
    Calculate the Out-of-Bag score for a bagging ensemble using PyTorch.
    
    Args:
        n_samples: Total number of samples in the dataset
        bootstrap_indices: List of lists containing indices used to train each estimator
        predictions: Tensor of shape (n_estimators, n_samples) with predictions from each estimator
        y_true: Tensor of shape (n_samples,) with true labels
    
    Returns:
        OOB accuracy score as a float
    """
    n_estimators = len(bootstrap_indices)
    
    # Create a boolean mask of OOB samples for each estimator
    # Shape: (n_estimators, n_samples)
    oob_mask = torch.zeros(n_estimators, n_samples, dtype=torch.bool)
    
    for i, indices in enumerate(bootstrap_indices):
        # Convert to tensor if needed
        if not isinstance(indices, torch.Tensor):
            indices = torch.tensor(indices, dtype=torch.long)
        # Mark samples that are NOT in bootstrap_indices as OOB
        oob_mask[i, indices] = False
        # All samples not in bootstrap_indices are OOB
        # We'll set all to True first, then set in-bag to False
        oob_mask[i, :] = True
        oob_mask[i, indices] = False
    
    # For each sample, collect predictions from estimators where it was OOB
    correct_predictions = 0
    total_oob_samples = 0
    
    for sample_idx in range(n_samples):
        # Get OOB predictions for this sample
        oob_predictions = predictions[oob_mask[:, sample_idx], sample_idx]
        
        # Skip if no OOB predictions
        if oob_predictions.numel() == 0:
            continue
        
        # Majority vote
        # Count votes for each class
        unique_classes = torch.unique(oob_predictions)
        vote_counts = torch.zeros(len(unique_classes), dtype=torch.long)
        
        for i, cls in enumerate(unique_classes):
            vote_counts[i] = torch.sum(oob_predictions == cls)
        
        # Get the class with most votes
        majority_vote = unique_classes[torch.argmax(vote_counts)]
        
        # Check if prediction matches true label
        if majority_vote == y_true[sample_idx]:
            correct_predictions += 1
        
        total_oob_samples += 1
    
    # Return accuracy or 0.0 if no OOB samples
    if total_oob_samples == 0:
        return 0.0
    
    return correct_predictions / total_oob_samples