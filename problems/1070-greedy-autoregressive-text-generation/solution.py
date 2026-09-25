import numpy as np

def generate_greedy(model, idx: list, max_new_tokens: int, context_size: int) -> list:
    idx = list(idx)

    for _ in range(max_new_tokens):
        # Crop to the last context_size tokens
        cropped = idx[-context_size:]

        # Convert to shape (1, T)
        input_array = np.array(cropped, dtype=np.int64)[None, :]

        # Get logits: shape (1, T, vocab_size)
        logits = model(input_array)

        # Logits for the last time step: shape (1, vocab_size)
        last_logits = logits[:, -1, :]

        # Greedy selection
        next_token = int(np.argmax(last_logits, axis=-1)[0])

        # Append to sequence
        idx.append(next_token)

    return idx