import numpy as np
from collections import Counter

def meteor_score(reference, candidate, alpha=0.9, beta=3, gamma=0.5):
    """
    Calculate METEOR score for machine translation evaluation using exact matching.
    
    Args:
        reference: Reference translation string
        candidate: Candidate translation string
        alpha: Weight for precision vs recall in F-mean (default 0.9)
        beta: Exponent for fragmentation penalty (default 3)
        gamma: Maximum penalty coefficient (default 0.5)
    
    Returns:
        METEOR score between 0 and 1
    """
    # Step 1: Tokenize and lower-case words
    ref_words = reference.lower().split()
    cand_words = candidate.lower().split()
    
    if not ref_words or not cand_words:
        return 0.0

    # Step 2: Compute unigram matches (handling duplicates correctly via intersection)
    ref_counts = Counter(ref_words)
    cand_counts = Counter(cand_words)
    
    overlap = ref_counts & cand_counts
    total_matches = sum(overlap.values())
    
    if total_matches == 0:
        return 0.0
        
    # Step 3: Calculate Precision, Recall, and F-mean
    precision = total_matches / len(cand_words)
    recall = total_matches / len(ref_words)
    
    # METEOR parameterized harmonic mean (heavily penalizes low recall if alpha=0.9)
    f_mean = (precision * recall) / (alpha * precision + (1 - alpha) * recall)
    
    # Step 4: Calculate Alignment and Chunks for Fragmentation Penalty
    # We find the specific indices of matched unigrams to determine structural chunks
    ref_matched_indices = []
    cand_remaining = list(cand_words)
    
    # Keep track of which candidate indices are matched to preserve ordering
    matched_pairs = []
    
    # Greedy alignment matching order of reference
    for r_idx, r_word in enumerate(ref_words):
        if r_word in cand_remaining:
            # Find the first occurrence in candidate and record it
            c_idx = cand_words.index(r_word)
            # Nullify this instance so it isn't matched twice
            cand_words[c_idx] = None 
            cand_remaining.remove(r_word)
            matched_pairs.append((r_idx, c_idx))
            
    # Sort matched pairs by candidate index to find contiguous chunks
    matched_pairs.sort(key=lambda pair: pair[1])
    
    # Count chunks: contiguous adjacent sequences in both candidate and reference
    chunks = 0
    if matched_pairs:
        chunks = 1
        for i in range(1, len(matched_pairs)):
            # If the current match is not immediately adjacent in both reference and candidate, 
            # it signals the start of a new structural chunk
            if matched_pairs[i][0] != matched_pairs[i-1][0] + 1:
                chunks += 1

    # Step 5: Final Score Calculation
    penalty = gamma * ((chunks / total_matches) ** beta)
    score = f_mean * (1 - penalty)
    
    return round(score, 3)