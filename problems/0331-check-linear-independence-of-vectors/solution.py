import torch

def is_linearly_independent(vectors: list[list[float]]) -> bool:
    """
    Check if a set of vectors is linearly independent.
    
    Args:
        vectors: List of vectors, where each vector is a list of floats.
                 All vectors must have the same dimension.
        
    Returns:
        True if vectors are linearly independent, False otherwise.
    """
    # Empty set is linearly independent by convention
    if len(vectors) == 0:
        return True

    # Validate that all vectors have the same dimension
    dim = len(vectors[0])
    for v in vectors:
        if len(v) != dim:
            raise ValueError("All vectors must have the same dimension")

    # Build a matrix whose rows are the vectors
    M = torch.tensor(vectors, dtype=torch.float64)

    # The vectors are linearly independent iff the rank equals the number
    # of vectors. If there are more vectors than dimensions, the rank can
    # never reach the number of vectors, so the result is automatically False.
    rank = torch.linalg.matrix_rank(M)
    return rank.item() == len(vectors)