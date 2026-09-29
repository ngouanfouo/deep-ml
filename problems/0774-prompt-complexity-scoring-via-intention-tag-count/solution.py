def score_prompt_complexity(prompts):
    """
    prompts: list of {'text': str, 'tags': list[str]}
    Returns: list of {'text': str, 'complexity': int} sorted by complexity desc (stable).
    """
    scored = [
        {'text': p['text'], 'complexity': len(set(p['tags']))}
        for p in prompts
    ]
    # Python's sort is stable, so ties keep original input order
    scored.sort(key=lambda x: x['complexity'], reverse=True)
    return scored