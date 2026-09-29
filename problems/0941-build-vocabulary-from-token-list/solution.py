def build_vocab(tokens):
    """
    Build a vocabulary dictionary from a list of tokens.

    Args:
        tokens: list of string tokens

    Returns:
        Dict mapping each unique token (sorted) to a unique integer ID starting from 0.
    """
    unique_tokens = sorted(set(tokens))
    return {token: idx for idx, token in enumerate(unique_tokens)}