import numpy as np

def td_lambda_error_bound(P, r, gamma, Phi, d, lambdas):
    """
    Compute and verify TD(lambda) asymptotic error bounds.
    
    Args:
        P: Transition probability matrix, shape (n, n)
        r: Expected reward vector, shape (n,)
        gamma: Discount factor (0 <= gamma < 1)
        Phi: Feature matrix, shape (n, k)
        d: Stationary distribution, shape (n,)
        lambdas: List of lambda values to evaluate
    
    Returns:
        Dictionary with keys:
            'true_values': list of floats
            'mc_error': float
            'td_errors': list of floats
            'ratios': list of floats
            'bound': float
            'bound_holds': list of bools
    """
    # Convert inputs to numpy arrays
    P = np.array(P, dtype=float)
    r = np.array(r, dtype=float)
    Phi = np.array(Phi, dtype=float)
    d = np.array(d, dtype=float)
    lambdas = np.array(lambdas, dtype=float)
    
    n = P.shape[0]
    k = Phi.shape[1]
    
    # 1. Compute the true value function V = (I - gamma*P)^(-1) * r
    I = np.eye(n)
    V = np.linalg.solve(I - gamma * P, r)
    true_values = V.tolist()
    
    # 2. Compute D matrix (diagonal of stationary distribution)
    D = np.diag(d)
    
    # 3. Compute Monte Carlo (MC) projection of V onto feature space
    Phi_T_D = Phi.T @ D
    A = Phi_T_D @ Phi
    b = Phi_T_D @ V
    w_mc = np.linalg.solve(A, b)
    V_mc = Phi @ w_mc
    
    # MC error: ||V - V_mc||_D
    diff_mc = V - V_mc
    mc_error = np.sqrt(diff_mc.T @ D @ diff_mc)
    
    # 4. For each lambda, compute TD(lambda) fixed-point solution and its error
    td_errors = []
    ratios = []
    
    for lam in lambdas:
        # TD(lambda) fixed-point solution:
        # w_td = (Phi^T D (I - lambda*gamma*P)^(-1) (I - gamma*P) Phi)^(-1) * Phi^T D (I - lambda*gamma*P)^(-1) r
        
        # Compute (I - lambda*gamma*P)
        I_minus_lambda_gamma_P = I - lam * gamma * P
        
        # Compute its inverse
        inv_I_minus_lambda_gamma_P = np.linalg.inv(I_minus_lambda_gamma_P)
        
        # Compute (I - gamma*P) Phi
        I_minus_gamma_P_Phi = (I - gamma * P) @ Phi
        
        # A_lambda = Phi^T D (I - lambda*gamma*P)^(-1) (I - gamma*P) Phi
        A_td = Phi_T_D @ inv_I_minus_lambda_gamma_P @ I_minus_gamma_P_Phi
        
        # b_lambda = Phi^T D (I - lambda*gamma*P)^(-1) r
        b_td = Phi_T_D @ inv_I_minus_lambda_gamma_P @ r
        
        # Solve for w_td
        w_td = np.linalg.solve(A_td, b_td)
        
        # Compute the TD(lambda) value function approximation
        V_td = Phi @ w_td
        
        # TD error: ||V - V_td||_D
        diff_td = V - V_td
        td_error = np.sqrt(diff_td.T @ D @ diff_td)
        td_errors.append(td_error)
        
        # Error amplification ratio
        # If mc_error is 0, the ratio should be 0 (since TD error will also be 0)
        if mc_error < 1e-10:
            ratio = 0.0
        else:
            ratio = td_error / mc_error
        ratios.append(ratio)
    
    # 5. Compute the theoretical upper bound
    bound = 1.0 / np.sqrt(1 - gamma**2)
    
    # 6. Check if the bound holds for each lambda
    # Convert to Python bool to avoid numpy bool_ type
    bound_holds = [bool(ratio <= bound + 1e-10) for ratio in ratios]
    
    # Round all numeric values to 4 decimal places
    return {
        'true_values': [round(v, 4) for v in true_values],
        'mc_error': round(float(mc_error), 4),
        'td_errors': [round(float(e), 4) for e in td_errors],
        'ratios': [round(float(r), 4) for r in ratios],
        'bound': round(float(bound), 4),
        'bound_holds': bound_holds
    }