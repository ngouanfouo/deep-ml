import numpy as np
import hashlib

def minhash_near_duplicates(documents: list[str], num_hashes: int, threshold: float, shingle_size: int, seed: int) -> list[tuple]:
    """
    Detect near-duplicate document pairs using MinHash.

    Returns a list of (i, j) index pairs (i < j) whose estimated Jaccard
    similarity meets or exceeds the given threshold.
    """
    p = 2**61 - 1
    
    # Generate hash function coefficients using numpy random generator
    rng = np.random.default_rng(seed)
    a = rng.integers(1, p, size=num_hashes)
    b = rng.integers(0, p, size=num_hashes)
    
    def get_shingles(doc: str) -> set:
        """Convert document to set of word-level k-shingles."""
        if not doc.strip():
            return set()
        
        # Lowercase and split into tokens
        tokens = doc.lower().split()
        
        if len(tokens) < shingle_size:
            # If fewer than shingle_size tokens, use entire sequence as one shingle
            return {''.join(tokens)}
        
        # Generate shingles of size shingle_size
        shingles = set()
        for i in range(len(tokens) - shingle_size + 1):
            shingle = ' '.join(tokens[i:i + shingle_size])
            shingles.add(shingle)
        return shingles
    
    def base_hash(shingle: str) -> int:
        """Compute base integer hash of a shingle using MD5."""
        md5_hash = hashlib.md5(shingle.encode('utf-8')).hexdigest()
        return int(md5_hash, 16) % p
    
    def hash_function(x: int, a_i: int, b_i: int) -> int:
        """Apply hash function h_i(x) = (a_i * x + b_i) mod p."""
        return (a_i * x + b_i) % p
    
    def compute_minhash_signature(shingles: set) -> np.ndarray:
        """Compute MinHash signature for a set of shingles."""
        if not shingles:
            # Empty shingle set: all entries are p
            return np.full(num_hashes, p, dtype=np.int64)
        
        # Compute base hashes for all shingles
        base_values = [base_hash(s) for s in shingles]
        
        # Initialize signature with large values
        signature = np.full(num_hashes, p, dtype=np.int64)
        
        # For each shingle, compute all hash values and take minimum
        for base_val in base_values:
            for i in range(num_hashes):
                h_val = hash_function(base_val, a[i], b[i])
                if h_val < signature[i]:
                    signature[i] = h_val
        
        return signature
    
    # Compute signatures for all documents
    signatures = []
    for doc in documents:
        shingles = get_shingles(doc)
        sig = compute_minhash_signature(shingles)
        signatures.append(sig)
    
    # Find near-duplicate pairs
    result = []
    n_docs = len(documents)
    
    for i in range(n_docs):
        for j in range(i + 1, n_docs):
            # Estimate Jaccard similarity
            sig_i = signatures[i]
            sig_j = signatures[j]
            
            # Count positions where signatures agree
            matches = np.sum(sig_i == sig_j)
            estimated_similarity = matches / num_hashes
            
            if estimated_similarity >= threshold:
                result.append((i, j))
    
    return result