import numpy as np

class MyReducer:
    """
    Implement your own dimensionality reduction to 10 dimensions.
    
    Your goal: Project high-dimensional data to 10 dimensions while
    preserving structure for classification.
    
    A k-NN classifier will be trained on your reduced data to evaluate quality.
    """
    
    def __init__(self):
        self.n_components = 10
        self.mean = None
        self.components = None
    
    def fit(self, X):
        """
        Learn the reduction from training data using PCA (via SVD).
        
        Args:
            X: Training data, shape (n_samples, n_features)
        
        Returns:
            self
        """
        # 1. Compute and store the mean for zero-centering
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean
        
        # 2. Use SVD for computational efficiency and numerical stability
        # X_centered = U * S * Vt
        _, _, Vt = np.linalg.svd(X_centered, full_matrices=False)
        
        # 3. Store the top 'n_components' principal axes (rows of Vt)
        self.components = Vt[:self.n_components]
        
        return self
    
    def transform(self, X):
        """
        Apply the learned reduction to data.
        
        Args:
            X: Data to transform, shape (n_samples, n_features)
        
        Returns:
            X_reduced: shape (n_samples, 10)
        """
        # 1. Center the data using the mean learned during fit()
        X_centered = X - self.mean
        
        # 2. Project the centered data onto the principal components
        # (n_samples, n_features) x (n_features, n_components)
        X_reduced = np.dot(X_centered, self.components.T)
        
        return X_reduced
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)