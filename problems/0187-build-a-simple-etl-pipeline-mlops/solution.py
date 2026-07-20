import torch
from typing import List, Tuple

def run_etl(csv_text: str) -> List[Tuple[str, float]]:
    """Run a simple ETL pipeline over CSV text with dynamic header matching.
    
    Uses PyTorch tensor operations for numerical aggregation.
    Returns a sorted list of (user_id, total_value) for event_type == "purchase".
    """
    # 1. EXTRACT & DYNAMIC HEADER PARSING
    lines = [line.strip() for line in csv_text.strip().split('\n') if line.strip()]
    if len(lines) <= 1:
        return []
        
    # Dynamically locate column indices from the header line
    header = [col.strip() for col in lines[0].split(',')]
    try:
        user_idx = header.index("user_id")
        event_idx = header.index("event_type")
        val_idx = header.index("value")
    except ValueError:
        # Fallback early if any essential column name is missing entirely
        return []
        
    valid_purchases = []
    
    # Process data rows using mapped dynamic offsets
    for line in lines[1:]:
        parts = [p.strip() for p in line.split(',')]
        if len(parts) <= max(user_idx, event_idx, val_idx):
            continue
            
        # Filter for purchase events
        if parts[event_idx] != "purchase":
            continue
            
        # Transform value string to float safely
        try:
            val = float(parts[val_idx])
            valid_purchases.append((parts[user_idx], val))
        except ValueError:
            continue
            
    if not valid_purchases:
        return []
        
    # 2. TRANSFORM (PyTorch Vector Aggregation)
    unique_users = sorted(list(set(u[0] for u in valid_purchases)))
    user_to_idx = {user_id: idx for idx, user_id in enumerate(unique_users)}
    
    indices = torch.tensor([user_to_idx[u[0]] for u in valid_purchases], dtype=torch.long)
    values = torch.tensor([u[1] for u in valid_purchases], dtype=torch.float64)
    
    output_tensor = torch.zeros(len(unique_users), dtype=torch.float64)
    output_tensor.index_add_(0, indices, values)
    
    # 3. LOAD
    total_values = output_tensor.tolist()
    return [(user_id, total_values[idx]) for user_id, idx in user_to_idx.items()]