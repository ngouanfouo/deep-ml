import torch

def byte_pair_encoding(corpus: dict, num_merges: int) -> list:
    """
    Train a BPE tokenizer on the given corpus using PyTorch for pair counting.
    
    Args:
        corpus: Dictionary mapping space-separated token sequences to their frequencies.
                Example: {"l o w </w>": 5, "n e w </w>": 6}
        num_merges: Number of merge operations to perform.
    
    Returns:
        List of tuples, where each tuple contains the two tokens that were merged.
        Example: [('l', 'o'), ('lo', 'w')]
    """
    # Parse the corpus into token lists with their frequencies
    tokenized_corpus = []
    for text, freq in corpus.items():
        tokens = text.split()
        tokenized_corpus.append((tokens, freq))
    
    merges = []
    
    for _ in range(num_merges):
        # Count all adjacent pairs across the corpus
        pair_counts = {}
        
        for tokens, freq in tokenized_corpus:
            if len(tokens) < 2:
                continue
            
            # Count each adjacent pair occurrence
            for i in range(len(tokens) - 1):
                pair = (tokens[i], tokens[i+1])
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        
        # If no pairs found, stop early
        if not pair_counts:
            break
        
        # Find the most frequent pair
        most_frequent_pair = max(pair_counts.items(), key=lambda x: x[1])[0]
        merges.append(most_frequent_pair)
        
        # Merge the pair everywhere it appears
        token1, token2 = most_frequent_pair
        merged_token = token1 + token2
        
        new_tokenized_corpus = []
        for tokens, freq in tokenized_corpus:
            new_tokens = []
            i = 0
            while i < len(tokens):
                # Check if we have a match for the pair
                if i < len(tokens) - 1 and tokens[i] == token1 and tokens[i+1] == token2:
                    new_tokens.append(merged_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_tokenized_corpus.append((new_tokens, freq))
        
        tokenized_corpus = new_tokenized_corpus
    
    return merges