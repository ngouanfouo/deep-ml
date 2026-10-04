import numpy as np

def self_play_train(payoff_matrix: list, initial_logits: list, num_iterations: int, learning_rate: float, sync_interval: int) -> dict:
    """
    Run a self-play training loop for a two-player zero-sum game.

    Args:
        payoff_matrix: 2D list of shape (n, n) - payoff matrix for player 1
        initial_logits: 1D list of shape (n,) - initial policy logits for both players
        num_iterations: int - number of training iterations
        learning_rate: float - learning rate for policy gradient ascent
        sync_interval: int - how often to sync opponent to current policy

    Returns:
        Dictionary with 'policy_logits', 'opponent_logits', 'policy_probs', 'expected_payoff'
    """
    def softmax(logits):
        # Numerically stable softmax
        shifted = logits - np.max(logits)
        exp_logits = np.exp(shifted)
        return exp_logits / np.sum(exp_logits)

    A = np.array(payoff_matrix, dtype=float)
    theta = np.array(initial_logits, dtype=float)
    theta_opp = np.array(initial_logits, dtype=float)

    for t in range(num_iterations):
        # Sync opponent periodically
        if t > 0 and sync_interval > 0 and t % sync_interval == 0:
            theta_opp = theta.copy()

        # Current policies
        pi = softmax(theta)
        pi_opp = softmax(theta_opp)

        # Expected payoff for each action of player 1 against opponent's mixed strategy
        q = A @ pi_opp

        # Expected value under player 1's current policy
        V = np.dot(pi, q)

        # Advantage and policy gradient
        advantage = q - V
        grad = pi * advantage

        # Gradient ascent update
        theta += learning_rate * grad

    # Final evaluation
    pi_final = softmax(theta)
    pi_opp_final = softmax(theta_opp)
    q_final = A @ pi_opp_final
    expected_payoff = np.dot(pi_final, q_final)

    return {
        'policy_logits': [round(float(x), 4) for x in theta],
        'opponent_logits': [round(float(x), 4) for x in theta_opp],
        'policy_probs': [round(float(x), 4) for x in pi_final],
        'expected_payoff': round(float(expected_payoff), 4)
    }