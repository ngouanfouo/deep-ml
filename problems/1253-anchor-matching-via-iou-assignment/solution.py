import numpy as np

def match_anchors(anchors, gt_boxes, pos_threshold=0.5, neg_threshold=0.4):
    """
    Assign each anchor a training label via IoU matching.

    Args:
        anchors: (N, 4) boxes in xyxy format [x1, y1, x2, y2]
        gt_boxes: (M, 4) ground-truth boxes in xyxy format
        pos_threshold: IoU >= this → positive (default 0.5)
        neg_threshold: IoU <  this → negative (default 0.4)

    Returns:
        labels:     (N,) int array with values {1=pos, 0=neg, -1=ignore}
        matched_gt: (N,) int array of matched GT index, or -1
    """
    anchors = np.asarray(anchors, dtype=float)
    gt_boxes = np.asarray(gt_boxes, dtype=float)
    
    # Edge case: N = 0
    if anchors.size == 0:
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=int)
    
    N = anchors.shape[0]
    
    # Edge case: M = 0 (no ground truth) → all anchors negative
    if gt_boxes.size == 0:
        return np.zeros((N,), dtype=int), -np.ones((N,), dtype=int)
    
    M = gt_boxes.shape[0]
    
    # --- Compute pairwise IoU matrix (N, M) ---
    # Broadcast anchors to (N, 1) and gt_boxes to (1, M)
    a_x1 = anchors[:, 0][:, None]
    a_y1 = anchors[:, 1][:, None]
    a_x2 = anchors[:, 2][:, None]
    a_y2 = anchors[:, 3][:, None]
    
    g_x1 = gt_boxes[:, 0][None, :]
    g_y1 = gt_boxes[:, 1][None, :]
    g_x2 = gt_boxes[:, 2][None, :]
    g_y2 = gt_boxes[:, 3][None, :]
    
    # Intersection
    inter_x1 = np.maximum(a_x1, g_x1)
    inter_y1 = np.maximum(a_y1, g_y1)
    inter_x2 = np.minimum(a_x2, g_x2)
    inter_y2 = np.minimum(a_y2, g_y2)
    
    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h  # (N, M)
    
    # Areas (clamp to >= 0 to handle degenerate boxes)
    area_a = np.maximum(0.0, a_x2 - a_x1) * np.maximum(0.0, a_y2 - a_y1)  # (N, 1)
    area_g = np.maximum(0.0, g_x2 - g_x1) * np.maximum(0.0, g_y2 - g_y1)  # (1, M)
    
    union = area_a + area_g - inter
    # Zero-area boxes / empty intersection → IoU 0
    iou = np.where(union > 0, inter / union, 0.0)  # (N, M)
    
    # --- Per-anchor best GT ---
    best_iou = np.max(iou, axis=1)     # (N,)
    best_gt = np.argmax(iou, axis=1)   # (N,)
    
    # --- Threshold labels ---
    labels = np.full((N,), -1, dtype=int)      # default: ignore
    matched_gt = np.full((N,), -1, dtype=int)
    
    pos_mask = best_iou >= pos_threshold
    neg_mask = best_iou < neg_threshold
    
    labels[pos_mask] = 1
    matched_gt[pos_mask] = best_gt[pos_mask]
    
    labels[neg_mask] = 0   # matched_gt stays -1 for negatives
    # Ignore anchors keep labels=-1, matched_gt=-1
    
    # --- Force a match for every GT ---
    # For each GT j, find the anchor with highest IoU against j
    best_anchor_per_gt = np.argmax(iou, axis=0)  # (M,)
    # Iterate in order; later GT index wins if several pick the same anchor
    for j in range(M):
        a_idx = best_anchor_per_gt[j]
        labels[a_idx] = 1
        matched_gt[a_idx] = j
    
    return labels, matched_gt