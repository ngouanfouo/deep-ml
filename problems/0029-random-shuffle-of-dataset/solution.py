import numpy as np


def shuffle_data(X, y, seed=None):
    """Shuffles datasets X and y randomly while maintaining the

    correspondence between features and labels.
    """
    if seed is not None:
        np.random.seed(seed)

    # Generate a shuffled permutation of indices based on the length of the dataset
    permutation = np.random.permutation(len(X))

    # Apply the same permutation to both arrays
    return X[permutation], y[permutation]