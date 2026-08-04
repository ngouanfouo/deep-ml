import torch

def detect_feature_drift(reference_data, production_data, num_bins: int = 10) -> dict:
    """
    Detect feature drift using Population Stability Index (PSI).
    """
    # Check for empty inputs
    if len(reference_data) == 0 or len(production_data) == 0:
        return {}
    
    # Convert to float tensors if needed
    if not isinstance(reference_data, torch.Tensor):
        reference_data = torch.tensor(reference_data, dtype=torch.float32)
    if not isinstance(production_data, torch.Tensor):
        production_data = torch.tensor(production_data, dtype=torch.float32)
    
    reference_data = reference_data.float()
    production_data = production_data.float()
    
    # Determine global bin edges based on combined data range
    combined_data = torch.cat([reference_data, production_data])
    min_val = torch.min(combined_data).item()
    max_val = torch.max(combined_data).item()
    
    # Add small padding if min == max to prevent zero range
    if max_val == min_val:
        min_val -= 0.5
        max_val += 0.5
    
    # Compute histograms using the full combined range
    ref_hist = torch.histc(reference_data, bins=num_bins, min=min_val, max=max_val)
    prod_hist = torch.histc(production_data, bins=num_bins, min=min_val, max=max_val)
    
    # Convert counts to proportions
    ref_prop = ref_hist / torch.sum(ref_hist)
    prod_prop = prod_hist / torch.sum(prod_hist)
    
    # Replace zero proportions with epsilon (0.0001)
    epsilon = 0.0001
    ref_prop = torch.where(ref_prop == 0, torch.tensor(epsilon), ref_prop)
    prod_prop = torch.where(prod_prop == 0, torch.tensor(epsilon), prod_prop)
    
    # Compute PSI: sum((prod_pct - ref_pct) * ln(prod_pct / ref_pct))
    psi = torch.sum((prod_prop - ref_prop) * torch.log(prod_prop / ref_prop)).item()
    
    # Determine drift level
    if psi < 0.1:
        drift_level = 'none'
        drift_detected = False
    elif psi < 0.25:
        drift_level = 'moderate'
        drift_detected = True
    else:
        drift_level = 'significant'
        drift_detected = True
    
    return {
        'psi': round(psi, 4),
        'drift_detected': drift_detected,
        'drift_level': drift_level
    }