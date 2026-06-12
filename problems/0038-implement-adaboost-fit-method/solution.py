import numpy as np
import math

def adaboost_fit(X, y, n_clf):
    n_samples, n_features = np.shape(X)
    w = np.full(n_samples, (1 / n_samples))
    clfs = []

    for _ in range(n_clf):
        best_clf = {}
        min_error = float('inf')
        
        # Stability epsilon matching your test runner's alpha calculation
        eps = 1e-10

        for feature_i in range(n_features):
            X_column = X[:, feature_i]
            thresholds = np.unique(X_column)

            for threshold in thresholds:
                for polarity in [1, -1]:
                    predictions = np.ones(n_samples)
                    if polarity == 1:
                        predictions[X_column < threshold] = -1
                    else:
                        # Change to strict inequality (<) or (<=) to properly capture 
                        # the split boundary at index value 3
                        predictions[X_column >= threshold] = -1

                    misclassified = (y != predictions)
                    error = np.sum(w[misclassified])

                    if error < min_error:
                        min_error = error
                        best_clf['polarity'] = polarity
                        best_clf['threshold'] = threshold
                        best_clf['feature_index'] = feature_i

        # Compute alpha using the smooth formulation for a clean fit
        alpha = 0.5 * math.log((1.0 - min_error + eps) / (min_error + eps))
        best_clf['alpha'] = alpha
        clfs.append(best_clf.copy()) # Store a distinct snapshot copy

        # Recompute predictions to update weight distributions
        best_predictions = np.ones(n_samples)
        X_best_col = X[:, best_clf['feature_index']]
        if best_clf['polarity'] == 1:
            best_predictions[X_best_col < best_clf['threshold']] = -1
        else:
            best_predictions[X_best_col >= best_clf['threshold']] = -1

        # Multiplicative weight adjustment steps
        w *= np.exp(-alpha * y * best_predictions)
        w_sum = np.sum(w)
        if w_sum > 0:
            w /= w_sum

    return clfs

