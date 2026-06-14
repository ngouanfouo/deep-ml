import numpy as np

def get_random_subsets(X, y, n_subsets, replacements=True):
    # Your code here
    n_samples=X.shape[0]
    subsets=[]

    if replacements:
        subset_size=n_samples
        for _ in range(n_subsets):
            indices=np.random.choice(n_samples, size=subset_size,replace=True)
            X_subset=X[indices].tolist()
            y_subset=y[indices].tolist()
            subsets.append((X_subset,y_subset))
    else:
        subset_size=n_samples//2
        for _ in range(n_subsets):

            indices=np.random.choice(n_samples,size=subset_size,replace=False)
            X_subset=X[indices].tolist()
            y_subset=y[indices].tolist()
            subsets.append((X_subset,y_subset))
    return subsets