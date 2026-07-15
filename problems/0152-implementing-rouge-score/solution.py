def rouge_1_score(reference: str, candidate: str) -> dict:
    """
    Compute ROUGE-1 score between reference and candidate texts.
    
    Returns a dictionary with precision, recall, and f1.
    """
    # Tokenize by splitting on whitespace
    ref_tokens = reference.lower().split()
    cand_tokens = candidate.lower().split()
    
    # Count token frequencies in reference and candidate
    ref_counts = {}
    for token in ref_tokens:
        ref_counts[token] = ref_counts.get(token, 0) + 1
    
    cand_counts = {}
    for token in cand_tokens:
        cand_counts[token] = cand_counts.get(token, 0) + 1
    
    # Calculate overlap: sum of min(count_ref, count_cand) for each word
    overlap = 0
    for token, ref_count in ref_counts.items():
        if token in cand_counts:
            overlap += min(ref_count, cand_counts[token])
    
    # Compute precision, recall, and F1
    ref_len = len(ref_tokens)
    cand_len = len(cand_tokens)
    
    if cand_len == 0:
        precision = 0.0
    else:
        precision = overlap / cand_len
    
    if ref_len == 0:
        recall = 0.0
    else:
        recall = overlap / ref_len
    
    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * (precision * recall) / (precision + recall)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1
    }