import numpy as np

def reward_model_validation(
    chosen_scores: np.ndarray,
    rejected_scores: np.ndarray,
    margin_thresholds: list[float]
) -> dict:
    """
    Compute validation metrics for a reward model on held-out preference pairs.

    Args:
        chosen_scores: 1D array of reward scores for preferred responses.
        rejected_scores: 1D array of reward scores for rejected responses.
        margin_thresholds: List of thresholds for margin-based accuracy.

    Returns:
        Dictionary with 'accuracy', 'mean_margin', 'concordance', 'margin_accuracy'.
    """
    # Compute margins (chosen - rejected)
    margins = chosen_scores - rejected_scores
    
    # Accuracy: fraction where chosen > rejected
    accuracy = np.mean(margins > 0)
    
    # Mean margin: average difference
    mean_margin = np.mean(margins)
    
    # Concordance: chosen > rejected counts as 1, ties as 0.5, losses as 0
    # Using np.where for vectorized computation
    concordance_scores = np.where(
        margins > 0,
        1.0,
        np.where(
            margins == 0,
            0.5,
            0.0
        )
    )
    concordance = np.mean(concordance_scores)
    
    # Margin accuracy: for each threshold, fraction where margin >= threshold
    margin_accuracy = {}
    for threshold in margin_thresholds:
        margin_accuracy[threshold] = np.mean(margins >= threshold)
    
    # Round all values to 4 decimal places
    return {
        'accuracy': round(float(accuracy), 4),
        'mean_margin': round(float(mean_margin), 4),
        'concordance': round(float(concordance), 4),
        'margin_accuracy': {
            threshold: round(float(value), 4) 
            for threshold, value in margin_accuracy.items()
        }
    }