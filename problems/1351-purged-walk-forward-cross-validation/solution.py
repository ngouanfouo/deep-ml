import numpy as np

def purged_walk_forward_splits(n_samples: int, n_splits: int, embargo: int) -> list:
    """Build expanding-window train/test splits with an embargo gap.

    Args:
        n_samples (int): total number of chronologically ordered samples.
        n_splits (int): number of test folds.
        embargo (int): samples immediately before each test block to drop from training.

    Returns:
        list[tuple[np.ndarray, np.ndarray]]: one (train_idx, test_idx) pair per fold.
    """
    fold_size = n_samples // (n_splits + 1)
    splits = []

    for k in range(1, n_splits + 1):
        test_start = k * fold_size
        test_end = n_samples if k == n_splits else (k + 1) * fold_size

        # Training covers everything before the test block minus the embargo gap.
        train_end = max(0, k * fold_size - embargo)

        train_idx = np.arange(0, train_end, dtype=np.int64)
        test_idx = np.arange(test_start, test_end, dtype=np.int64)

        splits.append((train_idx, test_idx))

    return splits