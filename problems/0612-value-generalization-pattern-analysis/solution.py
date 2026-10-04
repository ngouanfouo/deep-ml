import numpy as np

def value_generalization_analysis(
    features: np.ndarray,
    weights: np.ndarray,
    target_state: int,
    td_error: float,
    alpha: float
) -> dict:
    """
    Analyze how a semi-gradient TD update at a target state generalizes
    to value changes across all states under linear function approximation.
    """
    # Values before update: V(s) = features @ weights
    values_before = features @ weights

    # Semi-gradient TD update: w_new = w + alpha * td_error * x(target_state)
    x_target = features[target_state]
    weights_new = weights + alpha * td_error * x_target

    # Values after update
    values_after = features @ weights_new

    # Change in value for each state
    value_changes = values_after - values_before

    # Generalization ratio relative to the target state's value change
    target_change = value_changes[target_state]
    if target_change == 0:
        generalization_ratios = np.zeros_like(value_changes)
    else:
        generalization_ratios = value_changes / target_change

    # Round to 4 decimal places and convert to lists
    return {
        'values_before': np.round(values_before, 4).tolist(),
        'values_after': np.round(values_after, 4).tolist(),
        'value_changes': np.round(value_changes, 4).tolist(),
        'generalization_ratios': np.round(generalization_ratios, 4).tolist()
    }