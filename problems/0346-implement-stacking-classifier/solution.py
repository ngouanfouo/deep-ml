import numpy as np

def stacking_classifier(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
                        base_classifiers: list, meta_classifier, n_folds: int = 5) -> np.ndarray:
    """
    Implement a stacking classifier ensemble.
    
    Args:
        X_train: Training features of shape (n_samples, n_features)
        y_train: Training labels of shape (n_samples,)
        X_test: Test features of shape (m_samples, n_features)
        base_classifiers: List of classifier functions
        meta_classifier: Meta-level classifier function
        n_folds: Number of cross-validation folds
    
    Returns:
        np.ndarray: Final predictions on X_test as an array of integers
    """
    # Convert inputs to NumPy arrays to ensure indexing capabilities
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    n_samples = X_train.shape[0]
    n_base_clfs = len(base_classifiers)
    
    # -------------------------------------------------------------------------
    # STAGE 1: Generate Out-of-Fold (OOF) Training Meta-Features
    # -------------------------------------------------------------------------
    oof_meta_features = np.zeros((n_samples, n_base_clfs))
    fold_size = n_samples // n_folds
    
    for fold in range(n_folds):
        # Determine current sequential fold boundaries
        start_idx = fold * fold_size
        # The last fold absorbs any remaining samples due to integer division
        end_idx = n_samples if fold == n_folds - 1 else (fold + 1) * fold_size
        
        # Split into training indices and the current validation indices
        val_indices = np.arange(start_idx, end_idx)
        train_indices = np.setdiff1d(np.arange(n_samples), val_indices)
        
        X_fold_train = X_train[train_indices]
        y_fold_train = y_train[train_indices]
        X_fold_val = X_train[val_indices]
        
        # Get predictions from each base classifier for this validation fold
        for clf_idx, clf_func in enumerate(base_classifiers):
            fold_preds = clf_func(X_fold_train, y_fold_train, X_fold_val)
            oof_meta_features[val_indices, clf_idx] = fold_preds

    # -------------------------------------------------------------------------
    # STAGE 2: Generate Test Meta-Features from Full Training Data
    # -------------------------------------------------------------------------
    test_meta_features = np.zeros((X_test.shape[0], n_base_clfs))
    
    for clf_idx, clf_func in enumerate(base_classifiers):
        test_preds = clf_func(X_train, y_train, X_test)
        test_meta_features[:, clf_idx] = test_preds
        
    # -------------------------------------------------------------------------
    # STAGE 3: Final Meta-Classifier Predictions
    # -------------------------------------------------------------------------
    final_predictions = meta_classifier(oof_meta_features, y_train, test_meta_features)
    
    return np.array(final_predictions, dtype=int)