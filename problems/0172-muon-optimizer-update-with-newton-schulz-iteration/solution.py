import torch

def newtonschulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Apply the Newton-Schulz (quintic) iteration for 5 steps to matrix G using PyTorch.
    """
    # Muon's optimized magic coefficients
    a, b, c = 3.4445, -4.7750, 2.0315
    
    # Cast to float64 for high-precision matrix iteration
    X = G.to(torch.float64)
    M, N = X.shape
    if M > N:
        X = X.T
        
    # Spectral normalization
    X = X / (torch.linalg.matrix_norm(X) + eps)
    
    # Quintic iteration
    for _ in range(steps):
        A = torch.matmul(X, X.T)
        B = b * A + c * torch.matmul(A, A)
        X = a * X + torch.matmul(B, X)
        
    if M > N:
        X = X.T
        
    return X.to(G.dtype)

def muon_update(theta: torch.Tensor, grad: torch.Tensor, B_prev: torch.Tensor, mu: float, lr: float) -> tuple:
    """
    Performs one Muon optimizer update using PyTorch.
    """
    # Step 1: Momentum update
    B_new = mu * B_prev + grad
    
    # Step 2: Precondition with Newton-Schulz5
    O = newtonschulz5(B_new, steps=5)
    
    # Step 3: Update theta
    theta_new = theta - lr * O
    
    return theta_new, B_new, O