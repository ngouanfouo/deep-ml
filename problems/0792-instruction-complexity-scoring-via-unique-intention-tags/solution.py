def score_dialog_complexity(dialogs):
    """
    Args:
        dialogs: list of lists of intention tag strings
    Returns:
        list of (index, score) tuples sorted by score desc, then index asc
    """
    scored = [(i, len(set(tags))) for i, tags in enumerate(dialogs)]
    # Sort by score descending, then index ascending
    scored.sort(key=lambda x: (-x[1], x[0]))
    return scored