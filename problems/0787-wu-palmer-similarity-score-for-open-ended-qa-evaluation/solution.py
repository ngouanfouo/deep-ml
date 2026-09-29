def wups(depth_w1: int, depth_w2: int, depth_lca: int) -> float:
    """Compute Wu-Palmer Similarity between two words in a taxonomy."""
    denominator = depth_w1 + depth_w2
    if denominator == 0:
        return 0.0
    return (2.0 * depth_lca) / denominator