import numpy as np

class Node:
    """Decision tree node."""
    def __init__(self, is_leaf=True, value=None, feature=None, threshold=None, 
                 left=None, right=None):
        self.is_leaf = is_leaf
        self.value = value
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right

def decision_tree_regressor(X_train, y_train, X_test, max_depth=2, min_samples_split=2):
    """
    Build a decision tree for regression and predict on test data.
    
    Args:
        X_train: Training features, shape (n_samples, n_features)
        y_train: Training targets, shape (n_samples,)
        X_test: Test features, shape (m_samples, n_features)
        max_depth: Maximum depth of the tree
        min_samples_split: Minimum samples required to split a node
    
    Returns:
        List of predictions for X_test, rounded to 4 decimal places
    """
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    X_test = np.array(X_test)
    
    def build_tree(X, y, depth):
        """Recursively build the decision tree."""
        n_samples = len(y)
        
        # Check stopping conditions
        if depth >= max_depth or n_samples < min_samples_split or len(np.unique(y)) == 1:
            return Node(is_leaf=True, value=np.mean(y))
        
        # Find best split
        best_mse_reduction = -float('inf')
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        
        current_mse = np.mean((y - np.mean(y)) ** 2)
        n_features = X.shape[1]
        
        for feature_idx in range(n_features):
            feature_values = X[:, feature_idx]
            unique_values = np.unique(feature_values)
            
            if len(unique_values) <= 1:
                continue
            
            thresholds = (unique_values[:-1] + unique_values[1:]) / 2.0
            
            for threshold in thresholds:
                left_mask = feature_values <= threshold
                right_mask = ~left_mask
                
                if np.sum(left_mask) == 0 or np.sum(right_mask) == 0:
                    continue
                
                y_left = y[left_mask]
                y_right = y[right_mask]
                
                mse_left = np.mean((y_left - np.mean(y_left)) ** 2)
                mse_right = np.mean((y_right - np.mean(y_right)) ** 2)
                
                weight_left = len(y_left) / n_samples
                weight_right = len(y_right) / n_samples
                weighted_mse = weight_left * mse_left + weight_right * mse_right
                
                mse_reduction = current_mse - weighted_mse
                
                if mse_reduction > best_mse_reduction:
                    best_mse_reduction = mse_reduction
                    best_feature = feature_idx
                    best_threshold = threshold
                    best_left_mask = left_mask
                    best_right_mask = right_mask
        
        if best_feature is None:
            return Node(is_leaf=True, value=np.mean(y))
        
        left_child = build_tree(X[best_left_mask], y[best_left_mask], depth + 1)
        right_child = build_tree(X[best_right_mask], y[best_right_mask], depth + 1)
        
        return Node(is_leaf=False, feature=best_feature, threshold=best_threshold,
                    left=left_child, right=right_child)
    
    def predict(node, X):
        """Predict using the decision tree."""
        if node.is_leaf:
            return np.full(len(X), node.value)
        
        left_mask = X[:, node.feature] <= node.threshold
        right_mask = ~left_mask
        
        predictions = np.zeros(len(X))
        
        if np.sum(left_mask) > 0:
            predictions[left_mask] = predict(node.left, X[left_mask])
        if np.sum(right_mask) > 0:
            predictions[right_mask] = predict(node.right, X[right_mask])
        
        return predictions
    
    tree = build_tree(X_train, y_train, 0)
    predictions = predict(tree, X_test)
    
    return [round(float(pred), 4) for pred in predictions]