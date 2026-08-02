import numpy as np

def trpo_step(
    theta: np.ndarray,
    states: np.ndarray,
    actions: np.ndarray,
    advantages: np.ndarray,
    num_states: int,
    num_actions: int,
    delta: float,
    cg_iters: int = 10,
    line_search_steps: int = 10,
    line_search_decay: float = 0.5
) -> np.ndarray:
    """
    Perform a single TRPO policy update for a tabular softmax policy.
    """
    N = len(states)
    
    def get_policy(params: np.ndarray) -> np.ndarray:
        # Reshape to (S, A) for clean per-state softmax calculation
        logits = params.reshape(num_states, num_actions)
        shift_logits = logits - np.max(logits, axis=1, keepdims=True)
        exps = np.exp(shift_logits)
        return exps / np.sum(exps, axis=1, keepdims=True)

    def get_surrogate_objective(params: np.ndarray, pi_old: np.ndarray) -> float:
        pi_new = get_policy(params)
        # Select probabilities for actions actually executed in the batch
        p_old = pi_old[states, actions]
        p_new = pi_new[states, actions]
        ratios = p_new / p_old
        return float(np.mean(ratios * advantages))

    def get_kl(params: np.ndarray, pi_old: np.ndarray) -> float:
        pi_new = get_policy(params)
        # Empirical state visitation counts to weight the per-state KL
        state_counts = np.bincount(states, minlength=num_states)
        state_freqs = state_counts / N
        
        eps = 1e-15
        kl_per_state = np.sum(pi_old * (np.log(pi_old + eps) - np.log(pi_new + eps)), axis=1)
        return float(np.sum(state_freqs * kl_per_state))

    # 1. Compute current policy distribution snapshot
    pi_old = get_policy(theta)

    # 2. Compute the gradient of the surrogate objective at the current parameters
    # g = dL/dtheta evaluated at theta_old
    g = np.zeros((num_states, num_actions), dtype=float)
    for i in range(N):
        s, a, adv = states[i], actions[i], advantages[i]
        # Score function gradient for tabular softmax logit
        g[s, :] += adv * ((np.arange(num_actions) == a) - pi_old[s, :])
    g /= N
    g_flat = g.flatten()

    if np.allclose(g_flat, 0):
        return theta.copy()

    # Calculate empirical state distribution frequency weights
    state_counts = np.bincount(states, minlength=num_states)
    state_freqs = state_counts / N

    # 3. Define the Fisher Vector Product (FVP) linear operator function: F * v
    def fisher_vector_product(v_flat: np.ndarray, damping: float = 1e-8) -> np.ndarray:
        v = v_flat.reshape(num_states, num_actions)
        fvp = np.zeros_like(v)
        for s in range(num_states):
            if state_freqs[s] > 0:
                p = pi_old[s, :]
                # Analytical Fvp block form for tabular softmax: p * v - p * (p^T v)
                p_dot_v = np.dot(p, v[s, :])
                fvp[s, :] = state_freqs[s] * (p * v[s, :] - p * p_dot_v)
        # Add diagonal damping stabilization
        return fvp.flatten() + damping * v_flat

    # 4. Use Conjugate Gradient method to compute F^(-1) * g
    x = np.zeros_like(g_flat)
    r = g_flat.copy()
    p = g_flat.copy()
    rdotr = np.dot(r, r)
    
    for _ in range(cg_iters):
        fvp = fisher_vector_product(p)
        alpha = rdotr / np.dot(p, fvp)
        x += alpha * p
        r -= alpha * fvp
        new_rdotr = np.dot(r, r)
        if new_rdotr < 1e-10:
            break
        p = r + (new_rdotr / rdotr) * p
        rdotr = new_rdotr

    # 5. Compute step direction scale boundary limit
    xFx = np.dot(x, fisher_vector_product(x))
    if xFx <= 0:
        return theta.copy()
        
    beta = np.sqrt(2.0 * delta / xFx)
    full_step = beta * x

    # 6. Backtracking Line Search
    old_objective = get_surrogate_objective(theta, pi_old)
    
    for step_idx in range(line_search_steps):
        alpha = line_search_decay ** step_idx
        theta_candidate = theta + alpha * full_step
        
        candidate_objective = get_surrogate_objective(theta_candidate, pi_old)
        kl = get_kl(theta_candidate, pi_old)
        
        # Verify strict structural improvement and constraint satisfaction bounds
        if (candidate_objective > old_objective) and (kl <= delta):
            return theta_candidate

    # Return original vector if zero improvements could be found safely
    return theta.copy()