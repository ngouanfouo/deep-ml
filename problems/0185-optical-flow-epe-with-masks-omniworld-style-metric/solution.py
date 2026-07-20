from typing import Optional, Union

try:
    import numpy as np
except Exception:
    np = None

ArrayLike = Union[list, "np.ndarray"]

def flow_epe(pred: ArrayLike,
             gt: ArrayLike,
             mask: Optional[ArrayLike] = None,
             max_flow: Optional[float] = None) -> float:
    """
    Compute mean End-Point Error (EPE) between predicted and ground-truth optical flow.

    Args:
        pred, gt: (H, W, 2) lists or NumPy arrays.
        mask: optional (H, W) or broadcastable to (H, W); 1=include, 0=ignore.
        max_flow: optional float; clip per-pixel EPE to this value.

    Returns:
        float: mean EPE over valid pixels. Returns -1 on invalid input or if no valid pixels.
    """
    # 1. Fallback if numpy isn't loaded or available
    if np is None:
        return -1.0
        
    try:
        # Convert inputs to standard NumPy arrays
        pred_arr = np.array(pred, dtype=np.float64)
        gt_arr = np.array(gt, dtype=np.float64)
        
        # Validate spatial shape layout requirements (H, W, 2)
        if pred_arr.ndim != 3 or pred_arr.shape[-1] != 2 or pred_arr.shape != gt_arr.shape:
            return -1.0
            
        # 2. Compute the unclipped raw End-Point Error matrix across channels
        diff = pred_arr - gt_arr
        epe = np.linalg.norm(diff, axis=-1)
        
        # 3. Create structural evaluation validity mask (Ignore NaNs and Infinite values)
        valid_mask = np.isfinite(epe) & np.isfinite(pred_arr).all(axis=-1) & np.isfinite(gt_arr).all(axis=-1)
        
        # 4. Mix in the user-supplied spatial validation/occlusion mask if present
        if mask is not None:
            mask_arr = np.array(mask, dtype=np.float64)
            # Ensure the mask safely aligns or broadcasts with the spatial grid (H, W)
            if mask_arr.shape != epe.shape and mask_arr.shape != ():
                return -1.0
            valid_mask = valid_mask & (mask_arr > 0)
            
        # Terminate early with fallback code if no valid pixels pass conditions
        if not np.any(valid_mask):
            return -1.0
            
        # Isolate the remaining clean pixel errors
        valid_epe = epe[valid_mask]
        
        # 5. Apply the outlier clip parameter constraint if provided
        if max_flow is not None:
            if max_flow < 0:
                return -1.0
            valid_epe = np.clip(valid_epe, a_min=None, a_max=max_flow)
            
        # Return the final computed structural average scalar
        return float(np.mean(valid_epe))
        
    except (ValueError, TypeError, IndexError):
        # Gracefully handle dynamic array broadcasting mismatches or parsing types
        return -1.0