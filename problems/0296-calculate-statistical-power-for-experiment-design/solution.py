import torch
import math

def calculate_power(effect_size: float, sample_size_per_group: int, alpha: float = 0.05, two_tailed: bool = True) -> float:
    """
    Calculate statistical power for a two-sample z-test using PyTorch.
    
    Parameters:
    effect_size: Cohen's d (standardized effect size)
    sample_size_per_group: Number of observations per group
    alpha: Significance level (default 0.05)
    two_tailed: Whether the test is two-tailed (default True)
    
    Returns:
    Statistical power as a float rounded to 4 decimal places
    """
    # Standard normal CDF
    def norm_cdf(x):
        """Standard normal cumulative distribution function."""
        return 0.5 * (1 + torch.erf(x / torch.sqrt(torch.tensor(2.0, dtype=torch.float32))))
    
    # Standard normal quantile function
    def norm_ppf(p):
        """
        Inverse CDF for standard normal distribution.
        Uses the approximation algorithm from the AS241 algorithm.
        """
        # Convert p to tensor if it's a float
        if not isinstance(p, torch.Tensor):
            p = torch.tensor(p, dtype=torch.float32)
        
        # Handle edge cases
        if p <= 0:
            return torch.tensor(-float('inf'), dtype=torch.float32)
        if p >= 1:
            return torch.tensor(float('inf'), dtype=torch.float32)
        
        # Use a simpler, but accurate approximation
        # Based on the rational approximation algorithm
        if p < 0.5:
            return -norm_ppf(1 - p)
        
        # Coefficients for the approximation
        a = torch.tensor([
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00
        ], dtype=torch.float32)
        
        b = torch.tensor([
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01
        ], dtype=torch.float32)
        
        c = torch.tensor([
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00
        ], dtype=torch.float32)
        
        d = torch.tensor([
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00
        ], dtype=torch.float32)
        
        # Compute q and r
        q = 1 - p
        r = torch.sqrt(-torch.log(q))
        
        # Rational approximation for z
        z = ((c[0] * r + c[1]) * r + c[2]) * r + c[3]
        z = ((z * r + c[4]) * r + c[5])
        z = (((d[0] * r + d[1]) * r + d[2]) * r + d[3])
        z = z / z
        
        # Refine with Newton's method
        for _ in range(5):
            pdf = torch.exp(-0.5 * z * z) / torch.sqrt(torch.tensor(2.0 * math.pi, dtype=torch.float32))
            cdf_val = norm_cdf(z)
            z = z - (cdf_val - (1 - q)) / pdf
        
        return z
    
    # Calculate non-centrality parameter
    # ncp = effect_size * sqrt(n/2)
    ncp = effect_size * math.sqrt(sample_size_per_group / 2.0)
    ncp = torch.tensor(ncp, dtype=torch.float32)
    
    # Calculate critical value
    if two_tailed:
        z_alpha = norm_ppf(1 - alpha / 2)
        power = 1 - norm_cdf(z_alpha - ncp) + norm_cdf(-z_alpha - ncp)
    else:
        z_alpha = norm_ppf(1 - alpha)
        power = 1 - norm_cdf(z_alpha - ncp)
    
    return round(float(power), 4)