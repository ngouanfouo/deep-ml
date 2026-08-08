import numpy as np

def gaussian_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    """
    Implements Gaussian Naive Bayes classifier.
    
    Args:
        X_train: Training features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test features (shape: N_test x D)
    
    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    # Get number of features
    n_features = X_train.shape[1]
    n_test = X_test.shape[0]
    
    # Get unique classes
    classes = np.unique(y_train)
    n_classes = len(classes)
    
    # Epsilon for numerical stability
    eps = 1e-9
    
    # Pre-compute class parameters
    class_priors = np.zeros(n_classes)
    class_means = np.zeros((n_classes, n_features))
    class_variances = np.zeros((n_classes, n_features))
    
    for idx, cls in enumerate(classes):
        X_cls = X_train[y_train == cls]
        n_cls = len(X_cls)
        
        class_priors[idx] = n_cls / len(X_train)
        class_means[idx] = np.mean(X_cls, axis=0)
        class_variances[idx] = np.var(X_cls, axis=0) + eps
    
    # Initialize log posterior matrix: (n_test, n_classes)
    log_posteriors = np.zeros((n_test, n_classes))
    
    # Compute log likelihoods for all test samples
    # Using broadcasting for efficiency
    for idx in range(n_classes):
        mean = class_means[idx]
        variance = class_variances[idx]
        
        # Compute log prior (same for all test samples of this class)
        log_prior = np.log(class_priors[idx])
        
        # Compute log likelihood for each test sample
        # -0.5 * log(2*pi*variance) - (x - mean)^2 / (2*variance)
        # Broadcasting: X_test is (n_test, n_features), mean/variance are (n_features,)
        # Result: (n_test, n_features)
        log_likelihood = -0.5 * np.log(2 * np.pi * variance) - (X_test - mean)**2 / (2 * variance)
        
        # Sum across features: (n_test,)
        log_likelihood_sum = np.sum(log_likelihood, axis=1)
        
        # Log posterior: (n_test,)
        log_posteriors[:, idx] = log_prior + log_likelihood_sum
    
    # Select class with highest log posterior for each test sample
    predictions = classes[np.argmax(log_posteriors, axis=1)]
    
    return predictions