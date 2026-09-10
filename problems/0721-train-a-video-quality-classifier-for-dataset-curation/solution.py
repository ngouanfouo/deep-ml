import numpy as np

def curate_videos(train_features, train_labels, test_features):
    """
    Train a linear video quality classifier and predict keep/discard for test videos.

    Args:
        train_features: array-like of shape (N, D)
        train_labels: array-like of shape (N,), values in {0, 1}
        test_features: array-like of shape (M, D)

    Returns:
        list[int]: predictions of length M (1 = keep, 0 = discard)
    """
    # Convert to numpy arrays
    X = np.array(train_features, dtype=float)
    y = np.array(train_labels)
    X_test = np.array(test_features, dtype=float)
    
    # Handle empty training data
    if X.shape[0] == 0:
        return [0] * X_test.shape[0]
    
    # Split into classes
    X0 = X[y == 0]
    X1 = X[y == 1]
    
    # Compute class means
    mu0 = X0.mean(axis=0) if X0.shape[0] > 0 else np.zeros(X.shape[1])
    mu1 = X1.mean(axis=0) if X1.shape[0] > 0 else np.zeros(X.shape[1])
    
    # Compute within-class scatter matrix
    D = X.shape[1]
    S_W = np.zeros((D, D))
    if X0.shape[0] > 0:
        diff0 = X0 - mu0
        S_W += diff0.T @ diff0
    if X1.shape[0] > 0:
        diff1 = X1 - mu1
        S_W += diff1.T @ diff1
    
    # Compute Fisher direction using pseudo-inverse for robustness
    w = np.linalg.pinv(S_W) @ (mu1 - mu0)
    
    # Compute threshold as midpoint between projected class means
    proj0 = w @ mu0
    proj1 = w @ mu1
    threshold = (proj0 + proj1) / 2.0
    
    # Project test features and predict
    projections = X_test @ w
    predictions = (projections > threshold).astype(int).tolist()
    
    return predictions