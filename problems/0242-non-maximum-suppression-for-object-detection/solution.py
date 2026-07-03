import numpy as np

def non_maximum_suppression(boxes, scores, iou_threshold):
    """
    Apply Non-Maximum Suppression (NMS) to bounding boxes.
    
    Args:
        boxes: Array-like of shape (N, 4) with boxes in format [x1, y1, x2, y2]
        scores: Array-like of shape (N,) with confidence scores
        iou_threshold: float, IoU threshold for suppression (0 to 1)
    
    Returns:
        List of indices of kept boxes, ordered by descending score
        Returns -1 for invalid inputs
    """
    # Handle empty input
    if len(boxes) == 0 or len(scores) == 0:
        return []
    
    # Convert to numpy arrays
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Input validation
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        return -1
    if scores.ndim != 1:
        return -1
    if len(boxes) != len(scores):
        return -1
    if not (0 <= iou_threshold <= 1):
        return -1
    
    # Sort indices by score descending
    indices = np.argsort(scores)[::-1]
    
    # Keep track of kept indices
    keep = []
    
    while len(indices) > 0:
        # Select the highest score box
        current_idx = int(indices[0])  # Convert to Python int
        keep.append(current_idx)
        
        # Remove the selected index
        indices = indices[1:]
        
        # If no more boxes, break
        if len(indices) == 0:
            break
        
        # Get the current box
        current_box = boxes[current_idx]
        
        # Compute IoU with remaining boxes
        remaining_boxes = boxes[indices]
        
        # Compute intersection coordinates
        x1 = np.maximum(current_box[0], remaining_boxes[:, 0])
        y1 = np.maximum(current_box[1], remaining_boxes[:, 1])
        x2 = np.minimum(current_box[2], remaining_boxes[:, 2])
        y2 = np.minimum(current_box[3], remaining_boxes[:, 3])
        
        # Compute intersection area
        width = np.maximum(0, x2 - x1)
        height = np.maximum(0, y2 - y1)
        intersection = width * height
        
        # Compute areas
        current_area = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        remaining_areas = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
        
        # Compute union area
        union = current_area + remaining_areas - intersection
        
        # Compute IoU
        iou = intersection / union
        
        # Keep only boxes with IoU below threshold
        indices = indices[iou <= iou_threshold]
    
    return keep