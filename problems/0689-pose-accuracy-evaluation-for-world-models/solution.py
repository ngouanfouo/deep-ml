import numpy as np

def evaluate_pose_accuracy(pred_poses, gt_poses):
    """
    Evaluate accuracy of predicted camera poses against ground truth.
    
    Args:
        pred_poses: array-like of shape (N, 4, 4), predicted transformation matrices
        gt_poses: array-like of shape (N, 4, 4), ground-truth transformation matrices
    
    Returns:
        dict with 'mean_rotation_error', 'mean_translation_error',
             'median_rotation_error', 'median_translation_error'
        or -1 if inputs are invalid
    """
    try:
        pred = np.asarray(pred_poses, dtype=float)
        gt = np.asarray(gt_poses, dtype=float)
    except (ValueError, TypeError):
        return -1
    
    # Validate shapes
    if pred.ndim != 3 or gt.ndim != 3:
        return -1
    if pred.shape[1:] != (4, 4) or gt.shape[1:] != (4, 4):
        return -1
    if pred.shape[0] != gt.shape[0]:
        return -1
    if pred.shape[0] == 0:
        return -1
    
    # Validate finite values
    if not np.all(np.isfinite(pred)) or not np.all(np.isfinite(gt)):
        return -1
    
    N = pred.shape[0]
    
    # Extract rotation and translation components
    R_pred = pred[:, :3, :3]   # (N, 3, 3)
    t_pred = pred[:, :3, 3]    # (N, 3)
    R_gt = gt[:, :3, :3]       # (N, 3, 3)
    t_gt = gt[:, :3, 3]        # (N, 3)
    
    # --- Rotation error ---
    # Relative rotation: R_rel = R_pred^T @ R_gt
    R_rel = np.matmul(np.transpose(R_pred, (0, 2, 1)), R_gt)  # (N, 3, 3)
    # Trace of each relative rotation
    traces = np.trace(R_rel, axis1=1, axis2=2)  # (N,)
    # cos(theta) = (trace - 1) / 2, clamp to [-1, 1] for numerical stability
    cos_theta = np.clip((traces - 1.0) / 2.0, -1.0, 1.0)
    # Angle in radians, then convert to degrees
    rot_errors_deg = np.degrees(np.arccos(cos_theta))  # (N,)
    
    # --- Translation error ---
    trans_diffs = t_pred - t_gt                # (N, 3)
    trans_errors = np.linalg.norm(trans_diffs, axis=1)  # (N,)
    
    # --- Statistics ---
    result = {
        'mean_rotation_error': float(np.mean(rot_errors_deg)),
        'mean_translation_error': float(np.mean(trans_errors)),
        'median_rotation_error': float(np.median(rot_errors_deg)),
        'median_translation_error': float(np.median(trans_errors)),
    }
    
    return result