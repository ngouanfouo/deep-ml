import numpy as np

class NaiveBayes():
    def __init__(self, smoothing=1.0):
        self.smoothing = smoothing
        self.class_priors = None
        self.class_probs = None
        self.classes = None
        
    def forward(self, X, y):
        """
        Train the Bernoulli Naive Bayes classifier.
        
        Args:
            X: 2D NumPy array of binary features (n_samples, n_features)
            y: 1D NumPy array of class labels (n_samples,)
        """
        n_samples, n_features = X.shape
        
        # Get unique classes
        self.classes = np.unique(y)
        n_classes = len(self.classes)
        
        # Initialize probability arrays
        self.class_probs = np.zeros((n_classes, n_features))
        self.class_priors = np.zeros(n_classes)
        
        # For each class, compute prior and feature probabilities
        for i, cls in enumerate(self.classes):
            # Get samples belonging to this class
            mask = (y == cls)
            X_class = X[mask]
            n_class_samples = X_class.shape[0]
            
            # Compute class prior (no smoothing for prior)
            self.class_priors[i] = n_class_samples / n_samples
            
            # Compute feature probabilities with Laplace smoothing
            # P(feature=1 | class) = (count_ones + 1) / (n_class_samples + 2)
            feature_counts = X_class.sum(axis=0)  # Count of ones for each feature
            self.class_probs[i, :] = (feature_counts + self.smoothing) / (n_class_samples + 2.0 * self.smoothing)
    
    def predict(self, X):
        """
        Predict class labels for test data.
        
        Args:
            X: 2D NumPy array of binary features (n_samples, n_features)
            
        Returns:
            Predicted class labels as 1D NumPy array
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes)
        
        # Handle case where training data had only one class
        if n_classes == 1:
            return np.full(n_samples, self.classes[0])
        
        # Compute log posterior for each sample and each class
        log_posterior = np.zeros((n_samples, n_classes))
        
        for i, cls in enumerate(self.classes):
            # Log prior
            log_posterior[:, i] = np.log(self.class_priors[i])
            
            # For each feature, add log probability
            p_ones = self.class_probs[i, :]
            p_zeros = 1.0 - p_ones
            
            # Vectorized computation for all features at once
            # For each sample, compute sum of log probabilities
            log_posterior[:, i] += (X * np.log(p_ones) + (1 - X) * np.log(p_zeros)).sum(axis=1)
        
        # Predict class with highest log posterior
        predictions = self.classes[np.argmax(log_posterior, axis=1)]
        
        return predictions