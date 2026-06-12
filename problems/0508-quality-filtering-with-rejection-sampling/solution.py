import numpy as np

def quality_filter_rejection_sampling(scores: list, threshold: float, n_select: int = None) -> dict:
    """
    Filter generated samples using quality-based rejection sampling.
    
    Args:
        scores: list of float quality scores for generated candidate samples
        threshold: minimum quality score required for acceptance
        n_select: optional maximum number of samples to return (top by score)
    
    Returns:
        dict with 'accepted_indices', 'acceptance_rate', 'mean_quality'
    """
    total_samples = len(scores)
    if total_samples == 0:
        return {
            'accepted_indices': [],
            'acceptance_rate': 0.0,
            'mean_quality': 0.0
        }
    
    # 1. Track original indices and filter out samples below the threshold
    # Each element in passed_samples is a tuple: (original_index, score)
    passed_samples = [(idx, score) for idx, score in enumerate(scores) if score >= threshold]
    
    # 2. Compute the acceptance rate based on the threshold pass count
    # (Fraction of original samples that passed the threshold, before n_select truncation)
    acceptance_rate = round(len(passed_samples) / total_samples, 4)
    
    # 3. Rank the accepted samples by quality score in descending order
    passed_samples.sort(key=lambda item: item[1], reverse=True)
    
    # 4. If n_select is provided, truncate to keep only the top n_select samples
    if n_select is not None and n_select < len(passed_samples):
        final_samples = passed_samples[:n_select]
    else:
        final_samples = passed_samples
        
    # 5. Extract the final accepted indices and compute the mean quality
    if len(final_samples) > 0:
        accepted_indices = [item[0] for item in final_samples]
        mean_quality = round(float(np.mean([item[1] for item in final_samples])), 4)
    else:
        accepted_indices = []
        mean_quality = 0.0
        
    return {
        'accepted_indices': accepted_indices,
        'acceptance_rate': acceptance_rate,
        'mean_quality': mean_quality
    }
