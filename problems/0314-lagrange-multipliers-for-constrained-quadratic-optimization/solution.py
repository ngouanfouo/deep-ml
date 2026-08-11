import torch


def lagrange_optimize(
    Q: torch.Tensor, c: torch.Tensor, a: torch.Tensor, b: float
) -> dict:
    """Solve constrained quadratic optimization using Lagrange multipliers.

    Minimize: f(x) = (1/2) x^T Q x + c^T x
    Subject to: a^T x = b

    Args:
        Q: 2x2 symmetric positive definite matrix (torch.Tensor)
        c: 2-element vector (linear coefficients, torch.Tensor)
        a: 2-element vector (constraint coefficients, torch.Tensor)
        b: scalar (constraint value)

    Returns:
        Dictionary with 'x', 'lambda', and 'objective' keys
    """
    # Use float64 for optimal numerical precision
    Q = Q.to(torch.float64)
    c = c.to(torch.float64)
    a = a.to(torch.float64)
    b_val = float(b)

    # Construct the KKT matrix and right-hand side vector
    # [ Q   -a ] [ x ]   [ -c ]
    # [ -a^T 0 ] [ λ ] = [ -b ]
    K_top = torch.cat([Q, -a.reshape(-1, 1)], dim=1)
    K_bottom = torch.cat(
        [
            -a.reshape(1, -1),
            torch.tensor([[0.0]], dtype=torch.float64),
        ],
        dim=1,
    )
    K = torch.cat([K_top, K_bottom], dim=0)

    rhs = torch.cat([-c, torch.tensor([-b_val], dtype=torch.float64)])

    # Solve KKT system
    solution = torch.linalg.solve(K, rhs)

    # Extract x and lambda
    x_opt = solution[:2]
    lambda_opt = solution[2]

    # Compute objective value: f(x) = (1/2) x^T Q x + c^T x
    objective = 0.5 * (x_opt @ Q @ x_opt) + (c @ x_opt)

    # Return python primitive floats rounded cleanly to 4 decimal places
    return {
        "x": [round(x_opt[0].item(), 4), round(x_opt[1].item(), 4)],
        "lambda": round(lambda_opt.item(), 4),
        "objective": round(objective.item(), 4),
    }