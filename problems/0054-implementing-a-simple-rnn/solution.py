import numpy as np


def rnn_forward(
    input_sequence: list[list[float]],
    initial_hidden_state: list[float],
    Wx: list[list[float]],
    Wh: list[list[float]],
    b: list[float],
) -> list[float]:
    # Convert all lists into NumPy arrays for vector operations
    # Ensure hidden states and bias are column vectors shape (hidden_size, 1)
    h = np.array(initial_hidden_state).reshape(-1, 1)
    W_x = np.array(Wx)
    W_h = np.array(Wh)
    bias = np.array(b).reshape(-1, 1)

    # Process each step in the sequence sequentially
    for x in input_sequence:
        # Reshape current input vector to column vector: (input_size, 1)
        x_t = np.array(x).reshape(-1, 1)

        # Compute the next hidden state
        h = np.tanh(np.dot(W_x, x_t) + np.dot(W_h, h) + bias)

    # Flatten the final hidden state and round to 4 decimal places
    final_hidden_state = np.round(h.flatten(), 4).tolist()

    return final_hidden_state