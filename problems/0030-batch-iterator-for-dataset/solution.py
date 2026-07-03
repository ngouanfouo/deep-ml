import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    # Get number of samples
    n_samples = X.shape[0]
    batches = []
    
    # Iterate over batches
    for i in range(0, n_samples, batch_size):
        X_batch = X[i:i+batch_size]
        if y is not None:
            y_batch = y[i:i+batch_size]
            batches.append([X_batch.tolist(), y_batch.tolist()])
        else:
            batches.append(X_batch.tolist())
    
    return batches