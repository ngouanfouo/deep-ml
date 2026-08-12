import torch

def pairwise_preference_judge(comparisons: list, criteria_weights: dict, tie_threshold: float) -> dict:
    """
    Analyze pairwise comparisons between LLM responses using PyTorch tensor operations.
    
    Args:
        comparisons: List of comparison dicts with 'id', 'scores_a', 'scores_b'
        criteria_weights: Dict mapping criterion names to importance weights
        tie_threshold: Maximum difference to declare a tie
    
    Returns:
        Dict with 'results', 'win_rate_a', 'win_rate_b', 'tie_rate', 'avg_margin'
    """
    # Handle empty comparisons
    if not comparisons:
        return {
            'results': [],
            'win_rate_a': 0.0,
            'win_rate_b': 0.0,
            'tie_rate': 0.0,
            'avg_margin': 0.0
        }
    
    # Extract criteria names from the first comparison
    criteria_names = list(criteria_weights.keys())
    
    # Normalize weights so they sum to 1
    total_weight = sum(criteria_weights.values())
    normalized_weights = {c: w / total_weight for c, w in criteria_weights.items()}
    
    # Convert weights to tensor
    weights_tensor = torch.tensor([normalized_weights[c] for c in criteria_names], dtype=torch.float32)
    
    results = []
    total_margin = 0.0
    wins_a = 0
    wins_b = 0
    ties = 0
    
    for comp in comparisons:
        # Extract scores for each criterion
        scores_a = [comp['scores_a'].get(c, 0.0) for c in criteria_names]
        scores_b = [comp['scores_b'].get(c, 0.0) for c in criteria_names]
        
        # Convert to tensors
        scores_a_tensor = torch.tensor(scores_a, dtype=torch.float32)
        scores_b_tensor = torch.tensor(scores_b, dtype=torch.float32)
        
        # Compute weighted scores
        weighted_a = torch.dot(weights_tensor, scores_a_tensor).item()
        weighted_b = torch.dot(weights_tensor, scores_b_tensor).item()
        
        # Compute margin
        margin = abs(weighted_a - weighted_b)
        total_margin += margin
        
        # Determine winner
        if margin <= tie_threshold:
            winner = 'tie'
            ties += 1
        elif weighted_a > weighted_b:
            winner = 'A'
            wins_a += 1
        else:
            winner = 'B'
            wins_b += 1
        
        results.append({
            'id': comp['id'],
            'winner': winner,
            'margin': round(margin, 4)
        })
    
    n_comparisons = len(comparisons)
    
    # Compute rates
    win_rate_a = round(wins_a / n_comparisons, 4)
    win_rate_b = round(wins_b / n_comparisons, 4)
    tie_rate = round(ties / n_comparisons, 4)
    avg_margin = round(total_margin / n_comparisons, 4)
    
    return {
        'results': results,
        'win_rate_a': win_rate_a,
        'win_rate_b': win_rate_b,
        'tie_rate': tie_rate,
        'avg_margin': avg_margin
    }