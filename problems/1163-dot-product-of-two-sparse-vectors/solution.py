def sparse_dot_product(vec1, vec2):
    """
    Computes the dot product of two sparse vectors represented as plain Python lists.
    """
    return sum(a * b for a, b in zip(vec1, vec2) if a != 0 and b != 0)