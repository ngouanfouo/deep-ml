import torch

def detect_outliers_iqr(data: list[float], k: float = 1.5) -> dict:
    """
    Detect and remove outliers using the IQR method.
    
    Args:
        data: List of numerical values
        k: IQR multiplier for determining outlier bounds (default 1.5)
    
    Returns:
        Dictionary with 'cleaned_data', 'outlier_indices', 'lower_bound', 'upper_bound'
    """
    # Convert data to tensor
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # Compute Q1 (25th percentile) and Q3 (75th percentile)
    # Note: torch.quantile requires PyTorch 1.7+
    q1 = torch.quantile(data_tensor, 0.25)
    q3 = torch.quantile(data_tensor, 0.75)
    
    # Compute IQR
    iqr = q3 - q1
    
    # Compute lower and upper bounds
    lower_bound = q1 - k * iqr
    upper_bound = q3 + k * iqr
    
    # Find outliers
    outlier_mask = (data_tensor < lower_bound) | (data_tensor > upper_bound)
    outlier_indices = torch.where(outlier_mask)[0].tolist()
    
    # Cleaned data: keep only non-outliers
    cleaned_data = data_tensor[~outlier_mask].tolist()
    
    # Round to 4 decimal places
    cleaned_data = [round(val, 4) for val in cleaned_data]
    lower_bound_rounded = round(lower_bound.item(), 4)
    upper_bound_rounded = round(upper_bound.item(), 4)
    
    return {
        'cleaned_data': cleaned_data,
        'outlier_indices': outlier_indices,
        'lower_bound': lower_bound_rounded,
        'upper_bound': upper_bound_rounded
    }