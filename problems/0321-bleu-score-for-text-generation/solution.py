import torch
from collections import Counter

def bleu_score(candidate: list[str], references: list[list[str]], max_n: int = 4) -> float:
    """
    Calculate BLEU score for a candidate sentence against reference sentences
    using PyTorch tensor operations for numerical computations.
    
    Args:
        candidate: List of tokens in the candidate sentence
        references: List of reference sentences, each as a list of tokens
        max_n: Maximum n-gram order (default: 4)
    
    Returns:
        BLEU score between 0 and 1 as a Python float
    """
    # If candidate is empty, BLEU is 0
    if not candidate:
        return 0.0
    
    # Calculate modified n-gram precisions
    precisions = []
    for n in range(1, max_n + 1):
        # Get n-grams for candidate
        candidate_ngrams = [tuple(candidate[i:i+n]) for i in range(len(candidate) - n + 1)]
        if not candidate_ngrams:
            # If candidate is shorter than n, precision is 0
            precisions.append(0.0)
            continue
        
        # Count n-grams in candidate
        candidate_counts = Counter(candidate_ngrams)
        
        # Get max counts from references for each n-gram
        max_ref_counts = {}
        for ref in references:
            ref_ngrams = [tuple(ref[i:i+n]) for i in range(len(ref) - n + 1)]
            ref_counts = Counter(ref_ngrams)
            for ngram, count in ref_counts.items():
                max_ref_counts[ngram] = max(max_ref_counts.get(ngram, 0), count)
        
        # Calculate clipped counts
        clipped_count = sum(min(candidate_counts[ngram], max_ref_counts.get(ngram, 0)) 
                            for ngram in candidate_counts)
        
        # Modified n-gram precision
        total_count = len(candidate_ngrams)
        if total_count == 0:
            precisions.append(0.0)
        else:
            precisions.append(clipped_count / total_count)
    
    # If any precision is 0, BLEU is 0
    if any(p == 0.0 for p in precisions):
        return 0.0
    
    # Calculate geometric mean of precisions using PyTorch
    precisions_tensor = torch.tensor(precisions, dtype=torch.float32)
    # Use log sum and exp for numerical stability
    log_precisions = torch.log(precisions_tensor)
    geometric_mean = torch.exp(torch.mean(log_precisions))
    
    # Calculate brevity penalty
    candidate_len = len(candidate)
    
    # Find reference length closest to candidate length
    ref_lengths = [len(ref) for ref in references]
    if ref_lengths:
        # Find closest length, break ties by choosing shorter
        closest_len = min(ref_lengths, key=lambda x: (abs(x - candidate_len), x))
    else:
        closest_len = candidate_len
    
    # Brevity penalty
    if candidate_len > closest_len:
        bp = 1.0
    else:
        # Use PyTorch for exp computation
        bp = torch.exp(torch.tensor(1.0 - closest_len / candidate_len, dtype=torch.float32)).item()
    
    # Final BLEU score
    score = bp * geometric_mean.item()
    
    return float(score)