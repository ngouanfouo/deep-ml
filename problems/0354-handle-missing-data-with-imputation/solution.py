import torch

def impute_missing_data(data: torch.Tensor, strategy: str = 'mean') -> torch.Tensor:
    """
    Impute missing values in a 2D tensor using the specified strategy.

    Args:
        data: 2D torch.Tensor with missing values represented as float('nan')
        strategy: Imputation strategy - 'mean', 'median', or 'mode'

    Returns:
        2D torch.Tensor with missing values imputed
    """
    # Create a copy of the data to avoid modifying the original
    imputed_data = data.clone()
    
    n_rows, n_cols = imputed_data.shape
    
    for col in range(n_cols):
        # Extract column data
        col_data = imputed_data[:, col]
        
        # Find non-missing values
        non_missing_mask = ~torch.isnan(col_data)
        non_missing_values = col_data[non_missing_mask]
        
        # If all values are missing, skip this column
        if len(non_missing_values) == 0:
            continue
        
        # Compute imputation value based on strategy
        if strategy == 'mean':
            impute_value = torch.mean(non_missing_values)
        elif strategy == 'median':
            impute_value = torch.median(non_missing_values)
        elif strategy == 'mode':
            # Find the most frequent value
            unique_values, counts = torch.unique(non_missing_values, return_counts=True)
            max_count = torch.max(counts)
            # Get all values with the max count (handle ties)
            mode_values = unique_values[counts == max_count]
            # If ties, use the smallest value
            impute_value = torch.min(mode_values)
        else:
            raise ValueError(f"Unknown strategy: {strategy}. Use 'mean', 'median', or 'mode'.")
        
        # Replace missing values
        missing_mask = torch.isnan(col_data)
        imputed_data[missing_mask, col] = impute_value
    
    return imputed_data