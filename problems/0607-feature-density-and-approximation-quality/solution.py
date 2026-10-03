import numpy as np

def feature_density_approximation(
    train_states: np.ndarray,
    train_values: np.ndarray,
    test_states: np.ndarray,
    test_values: np.ndarray,
    n_centers: int,
    sigma: float
) -> dict:
    train_states = np.asarray(train_states, dtype=float)
    train_values = np.asarray(train_values, dtype=float)
    test_states = np.asarray(test_states, dtype=float)
    test_values = np.asarray(test_values, dtype=float)

    # 1. Place RBF centers uniformly in [0, 1]
    if n_centers == 1:
        centers = np.array([0.5])
    else:
        centers = np.linspace(0.0, 1.0, n_centers)

    # 2. RBF feature map: phi_i(s) = exp(-(s - c_i)^2 / (2 * sigma^2))
    def rbf_features(states):
        diff = states[:, None] - centers[None, :]   # (n_states, n_centers)
        return np.exp(-(diff ** 2) / (2.0 * sigma ** 2))

    Phi_train = rbf_features(train_states)
    Phi_test = rbf_features(test_states)

    # 3. Least-squares fit of the weights
    weights, *_ = np.linalg.lstsq(Phi_train, train_values, rcond=None)

    # 4. Predictions and RMSE on the test set
    predictions = Phi_test @ weights
    rmse = np.sqrt(np.mean((predictions - test_values) ** 2))

    # 5. Round everything to 4 decimal places
    return {
        "centers": [round(float(c), 4) for c in centers],
        "weights": [round(float(w), 4) for w in weights],
        "predictions": [round(float(p), 4) for p in predictions],
        "rmse": round(float(rmse), 4),
    }