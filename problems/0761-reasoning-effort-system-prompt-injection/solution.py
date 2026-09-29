def inject_reasoning_effort(messages, effort_level):
    """
    messages: list of dicts with keys 'role' and 'content'
    effort_level: one of 'low', 'medium', 'high', 'max'
    Returns: new list of message dicts with reasoning-effort system prompt injected.
    """
    instructions = {
        "low": "Reasoning effort: low. Provide concise answers with minimal deliberation.",
        "medium": "Reasoning effort: medium. Think step by step before answering.",
        "high": "Reasoning effort: high. Carefully decompose the problem and verify each step.",
        "max": ("Reasoning effort: max. Exhaustively decompose the problem, "
                "stress-test all edge cases, and document every intermediate step."),
    }

    if effort_level not in instructions:
        raise ValueError(f"Unsupported effort level: {effort_level}")

    instruction = instructions[effort_level]

    # Make a fresh copy of every message dict (so no mutation of inputs)
    new_messages = [dict(m) for m in messages]

    if new_messages and new_messages[0].get("role") == "system":
        existing = new_messages[0].get("content", "")
        new_messages[0]["content"] = instruction + "\n" + existing
    else:
        new_messages.insert(0, {"role": "system", "content": instruction})

    return new_messages