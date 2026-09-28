import numpy as np

def td_backprop_update(state, next_state, reward, gamma, alpha, W1, b1, W2, b2, terminal=False):
    """
    Perform one semi-gradient TD(0) update step with a neural network value function.
    """
    state = np.asarray(state, dtype=float)
    next_state = np.asarray(next_state, dtype=float)
    W1 = np.array(W1, dtype=float)
    b1 = np.array(b1, dtype=float)
    W2 = np.array(W2, dtype=float)
    b2 = np.array(b2, dtype=float)

    # 1. Forward pass for current state s
    z1 = state @ W1 + b1
    a1 = np.maximum(0.0, z1)
    v_s = float((a1 @ W2 + b2).item())

    # 2. Forward pass for next state s'
    if terminal:
        v_next = 0.0
    else:
        z1_next = next_state @ W1 + b1
        a1_next = np.maximum(0.0, z1_next)
        v_next = float((a1_next @ W2 + b2).item())

    # 3. TD error
    td_error = reward + gamma * v_next - v_s

    # 4. Gradients (semi-gradient: only V(s) is differentiated)
    grad_W2 = a1.reshape(-1, 1)
    grad_b2 = np.array([1.0])

    relu_mask = (z1 > 0.0).astype(float)
    grad_z1 = W2.squeeze(-1) * relu_mask

    grad_W1 = np.outer(state, grad_z1)
    grad_b1 = grad_z1

    # 5. Semi-gradient update
    W1_new = W1 + alpha * td_error * grad_W1
    b1_new = b1 + alpha * td_error * grad_b1
    W2_new = W2 + alpha * td_error * grad_W2
    b2_new = b2 + alpha * td_error * grad_b2

    # Round to suppress float64 representation noise
    return (
        round(float(td_error), 4),
        np.round(W1_new, 10),
        np.round(b1_new, 10),
        np.round(W2_new, 10),
        np.round(b2_new, 10),
    )