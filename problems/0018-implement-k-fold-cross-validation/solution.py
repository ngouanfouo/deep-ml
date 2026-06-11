import numpy as np
from typing import List, Tuple

def k_fold_cross_validation(n_samples: int, k: int = 5, shuffle: bool = True) -> List[Tuple[List[int], List[int]]]:
    """
    Generate train/test index splits for k-fold cross-validation.
    
    Args:
        n_samples: Total number of samples in the dataset
        k: Number of folds (default 5)
        shuffle: Whether to shuffle indices before splitting (default True)
    
    Returns:
        List of (train_indices, test_indices) tuples
    """
    # Create the initial array of indices
    indices = np.arange(n_samples)
    
    # Shuffle indices if requested
    if shuffle:
        np.random.shuffle(indices)
        
    # Determine basic fold sizes and the remainder to distribute
    base_size = n_samples // k
    remainder = n_samples % k
    
    # Pre-calculate the split boundaries
    splits = []
    current_idx = 0
    
    for i in range(k):
        # Add 1 extra sample to the first 'remainder' folds
        fold_size = base_size + (1 if i < remainder else 0)
        start, end = current_idx, current_idx + fold_size
        splits.append((start, end))
        current_idx = end
        
    # Generate the train/test index splits
    results = []
    for start, end in splits:
        # The current slice becomes the test set
        test_indices = indices[start:end].tolist()
        
        # Everything else becomes the training set
        train_indices = np.concatenate([indices[:start], indices[end:]]).tolist()
        
        results.append((train_indices, test_indices))
        
    return results

