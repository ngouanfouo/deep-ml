def fim_transform(tokens: list, i: int, j: int) -> list:
    """
    Apply Fill-in-the-Middle (PSM) transformation to a token sequence.
    """
    prefix = tokens[:i]
    middle = tokens[i:j]
    suffix = tokens[j:]

    return ['<PRE>'] + list(prefix) + ['<SUF>'] + list(suffix) + ['<MID>'] + list(middle) + ['<EOT>']