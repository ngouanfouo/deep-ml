import numpy as np

def minimax_with_learned_value(node, children, terminal_values, features, weights, depth, maximizing):
    """
    Perform depth-limited minimax search using a learned linear value function
    for leaf evaluation.
    """
    # Terminal nodes always return their exact value
    if node in terminal_values:
        return terminal_values[node]

    child_list = children.get(node, [])

    # If depth limit reached or no children, use learned value function
    if depth == 0 or not child_list:
        return float(np.dot(features[node], weights))

    if maximizing:
        best_value = -np.inf
        for child in child_list:
            value = minimax_with_learned_value(
                child, children, terminal_values, features, weights,
                depth - 1, False
            )
            if value > best_value:
                best_value = value
        return best_value
    else:
        best_value = np.inf
        for child in child_list:
            value = minimax_with_learned_value(
                child, children, terminal_values, features, weights,
                depth - 1, True
            )
            if value < best_value:
                best_value = value
        return best_value