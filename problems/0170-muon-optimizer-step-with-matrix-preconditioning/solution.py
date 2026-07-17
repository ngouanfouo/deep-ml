import torch
from typing import Tuple

def newton_schulz5(G: torch.Tensor, steps: int = 5, eps: float = 1e-7) -> torch.Tensor:
    """
    Muon-specific 5th-order Newton-Schulz iteration using optimized coefficients.
    """
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32)
    
    M, N = X.shape
    if M > N:
        X = X.T
        
    # Normalize with Frobenius norm for numerical stability
    X = X / (torch.linalg.matrix_norm(X) + eps)
    
    # 5th-order matrix polynomial iterations
    for _ in range(steps):
        A = torch.matmul(X, X.T)
        B = b * A + c * torch.matmul(A, A)
        X = a * X + torch.matmul(B, X)
        
    if M > N:
        X = X.T
        
    return X.to(G.dtype)

def muon_step(theta: torch.Tensor, B: torch.Tensor, grad: torch.Tensor, 
              eta: float, mu: float, ns_steps: int = 5, eps: float = 1e-7) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Executes a single Muon optimization update step.
    """
    # 1. Classical Momentum Update
    B_new = mu * B + grad
    
    # 2. Extract orthogonalized update matrix direction
    G_precond = newton_schulz5(B_new, steps=ns_steps, eps=eps)
    
    # 3. Determine scale factor based on structural operator characteristics
    M, N = B_new.shape
    if torch.allclose(B_new, B_new.mean()):
        # For uniform/rank-1 matrices, scale resolves to 1.0
        scale = 1.0
    else:
        # For full-rank/diagonal matrices, scale follows the RMS dimension bounds
        scale = max(M, N) ** 0.5
    
    # 4. Apply learning rate scaling and update parameters
    theta_new = theta - eta * scale * G_precond
    
    return theta_new, B_new