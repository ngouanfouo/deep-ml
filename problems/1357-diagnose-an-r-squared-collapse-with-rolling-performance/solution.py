import numpy as np


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Coefficient of determination; returns 0.0 when the variance of y_true is 0."""
    yt = np.asarray(y_true, dtype=float)
    yp = np.asarray(y_pred, dtype=float)
    
    ss_res = np.sum((yt - yp) ** 2)
    ss_tot = np.sum((yt - np.mean(yt)) ** 2)
    
    if ss_tot == 0.0:
        return 0.0
        
    return float(1.0 - (ss_res / ss_tot))


def rolling_r2(y_true: np.ndarray, y_pred: np.ndarray, window: int) -> np.ndarray:
    """r2_score over each contiguous window.

    Returns:
        np.ndarray: length len(y_true) - window + 1.
    """
    yt_arr = np.asarray(y_true, dtype=float)
    yp_arr = np.asarray(y_pred, dtype=float)
    n = len(yt_arr)
    
    if window <= 0 or window > n:
        return np.array([], dtype=float)
        
    out_len = n - window + 1
    res = np.empty(out_len, dtype=float)
    
    for i in range(out_len):
        yt_win = yt_arr[i : i + window]
        yp_win = yp_arr[i : i + window]
        res[i] = r2_score(yt_win, yp_win)
        
    return res