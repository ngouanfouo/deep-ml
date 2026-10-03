import numpy as np

def tile_coding_step(base_alpha: float, n_tilings: int, weights: list, active_tiles: list, target: float) -> dict:
    """
    Perform a single value function update using tile coding with adjusted step size.
    """
    weights = np.array(weights, dtype=float)

    # Adjust step size for the number of active features (one per tiling)
    adjusted_alpha = base_alpha / n_tilings

    # Current estimate: sum of weights at active tiles
    prediction_before = float(np.sum(weights[active_tiles]))

    # Prediction error
    error = target - prediction_before

    # Update only the active tile weights
    weights[active_tiles] += adjusted_alpha * error

    # New estimate after update
    prediction_after = float(np.sum(weights[active_tiles]))

    return {
        "adjusted_alpha": round(float(adjusted_alpha), 4),
        "prediction_before": round(prediction_before, 4),
        "prediction_after": round(prediction_after, 4),
        "updated_weights": [round(float(w), 4) for w in weights],
    }