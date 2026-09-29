def manage_reasoning_context(messages: list[dict], mode: str) -> list[dict]:
    """
    Apply conditional reasoning-trace retention to a message history.
    """
    # Deep-ish copy: new list with copied dicts (so we don't mutate input dicts)
    result = [dict(m) for m in messages]

    if mode == 'tool_calling':
        return result

    if mode == 'conversational':
        # Find the index of the last 'user' message
        last_user_idx = -1
        for i, m in enumerate(result):
            if m.get('role') == 'user':
                last_user_idx = i

        # If there is no user message, keep all reasoning
        if last_user_idx == -1:
            return result

        # Strip reasoning from assistant messages BEFORE last_user_idx
        for i, m in enumerate(result):
            if i < last_user_idx and m.get('role') == 'assistant' and 'reasoning' in m:
                # Preserve key order by rebuilding without 'reasoning'
                new_m = {k: v for k, v in m.items() if k != 'reasoning'}
                result[i] = new_m

        return result

    # Unknown mode -> return copy unchanged
    return result