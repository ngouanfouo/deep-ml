def lambda_return_study(rewards: list, values: list, gamma: float, lambdas: list) -> list:
    """
    Compute lambda-returns for each time step across multiple lambda values.
    
    Uses the recursive formulation:
        G_t^λ = r_t + γ [ (1 - λ) V(s_{t+1}) + λ G_{t+1}^λ ]
    with V(s_T) = 0 and G_T^λ = 0 for the terminal state after the last reward.
    """
    T = len(rewards)
    results = []

    for lam in lambdas:
        G = [0.0] * T
        G_next = 0.0          # G_{t+1}^λ, starts as G_T = 0
        V_next = 0.0          # V(s_{t+1}), starts as V(s_T) = 0

        for t in range(T - 1, -1, -1):
            G[t] = rewards[t] + gamma * ((1.0 - lam) * V_next + lam * G_next)
            # For the next iteration (t-1), the "next state" is s_t
            V_next = values[t]
            G_next = G[t]

        results.append([round(g, 4) for g in G])

    return results