import numpy as np
from scipy import stats

def descriptive_statistics(data: list | np.ndarray) -> dict:
    """
    Calculate various descriptive statistics metrics for a given dataset.
    
    Args:
        data: List or numpy array of numerical values
    
    Returns:
        Dictionary containing mean, median, mode, variance, standard deviation,
        percentiles (25th, 50th, 75th), and interquartile range (IQR)
    """
    # Convert input structure securely to a 1D NumPy float array
    arr = np.asarray(data, dtype=np.float64).flatten()
    
    if arr.size == 0:
        raise ValueError("Cannot calculate statistics on an empty dataset.")

    # Calculate basic tendencies
    mean_val = np.mean(arr)
    median_val = np.median(arr)
    
    # Calculate mode (handles multi-modal datasets by selecting the smallest first value)
    mode_res = stats.mode(arr, keepdims=True)
    mode_val = mode_res.mode[0]
    
    # Population variance and standard deviation (divide by N, ddof=0)
    variance_val = np.var(arr, ddof=0)
    std_val = np.std(arr, ddof=0)
    
    # Quantiles/Percentiles calculation using standard linear interpolation method
    p25 = np.percentile(arr, 25)
    p50 = np.percentile(arr, 50)
    p75 = np.percentile(arr, 75)
    iqr_val = p75 - p25
    
    # Construct metrics mapping dictionary rounded systematically to 4 decimals
    return {
        'mean': round(float(mean_val), 4),
        'median': round(float(median_val), 4),
        'mode': int(mode_val) if mode_val.is_integer() else round(float(mode_val), 4),
        'variance': round(float(variance_val), 4),
        'standard_deviation': round(float(std_val), 4),
        '25th_percentile': round(float(p25), 4),
        '50th_percentile': round(float(p50), 4),
        '75th_percentile': round(float(p75), 4),
        'interquartile_range': round(float(iqr_val), 4)
    }