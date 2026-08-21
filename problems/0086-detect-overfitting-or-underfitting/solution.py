import torch

def model_fit_quality(training_accuracy: torch.Tensor, test_accuracy: torch.Tensor) -> torch.Tensor:
    """
    Determine whether a machine learning model is overfitting, underfitting, or performing well
    based on training and test accuracy values.

    Args:
        training_accuracy (torch.Tensor): A scalar tensor representing the accuracy on training data (0 to 1).
        test_accuracy (torch.Tensor): A scalar tensor representing the accuracy on test data (0 to 1).

    Returns:
        torch.Tensor: A scalar tensor containing 1 (overfitting), -1 (underfitting), or 0 (good fit).
    """
    # Extract scalar values from tensors
    train_acc = training_accuracy.item() if torch.is_tensor(training_accuracy) else training_accuracy
    test_acc = test_accuracy.item() if torch.is_tensor(test_accuracy) else test_accuracy
    
    if train_acc - test_acc > 0.2:
        return torch.tensor(1)
    elif train_acc < 0.7 and test_acc < 0.7:
        return torch.tensor(-1)
    else:
        return torch.tensor(0)