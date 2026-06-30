import numpy as np
import math

# scipy is not available in this environment. The two helpers below compute
# the regularized incomplete beta function via the Numerical Recipes
# continued-fraction algorithm — you'll need them to compute the p-value
# from the t-distribution.
def _betacf(a, b, x):
    """Continued fraction for the incomplete beta function."""
    max_iter, eps = 200, 3e-12
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < 1e-30: d = 1e-30
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < 1e-30: d = 1e-30
        c = 1.0 + aa / c
        if abs(c) < 1e-30: c = 1e-30
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps: break
    return h

def _betainc(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(math.log(x) * a + math.log(1.0 - x) * b - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b

def _t_cdf(t, df):
    """
    Compute the CDF of the t-distribution at value t with df degrees of freedom.
    """
    if t == 0:
        return 0.5
    x = df / (df + t * t)
    # For t > 0, use the regularized beta function
    return 1 - 0.5 * _betainc(df / 2, 0.5, x)

def _t_2tail_pvalue(t, df):
    """
    Compute the two-tailed p-value for t-statistic with df degrees of freedom.
    """
    # For negative t, use symmetry
    t_abs = abs(t)
    # One-tailed p-value
    if t_abs == 0:
        return 1.0
    p_one_tail = 1 - _t_cdf(t_abs, df)
    # Two-tailed p-value
    p_two_tail = 2 * p_one_tail
    return min(p_two_tail, 1.0)

def two_sample_t_test(sample1: list[float], sample2: list[float],
                      alpha: float = 0.05) -> dict:
    """
    Perform a two-sample independent t-test (Welch's t-test).

    Args:
        sample1: First sample data
        sample2: Second sample data
        alpha: Significance level (default 0.05)

    Returns:
        Dictionary containing:
        - t_statistic: The calculated t-statistic
        - p_value: Two-tailed p-value
        - degrees_of_freedom: Degrees of freedom (Welch-Satterthwaite)
        - reject_null: Boolean, whether to reject null hypothesis
        - cohens_d: Effect size (Cohen's d)
    """
    # Convert to numpy arrays for easier computation
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    
    n1 = len(sample1)
    n2 = len(sample2)
    
    # Calculate means
    mean1 = np.mean(sample1)
    mean2 = np.mean(sample2)
    mean_diff = mean1 - mean2
    
    # Calculate variances (unbiased, ddof=1)
    var1 = np.var(sample1, ddof=1)
    var2 = np.var(sample2, ddof=1)
    
    # Calculate standard error of the difference
    se = np.sqrt(var1 / n1 + var2 / n2)
    
    # Calculate t-statistic
    t_statistic = mean_diff / se
    
    # Calculate degrees of freedom (Welch-Satterthwaite)
    df = (var1 / n1 + var2 / n2) ** 2 / (
        (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
    )
    
    # Calculate two-tailed p-value
    p_value = _t_2tail_pvalue(t_statistic, df)
    
    # Decision: reject or fail to reject null hypothesis
    reject_null = p_value < alpha
    
    # Calculate Cohen's d (pooled standard deviation)
    # For unequal variances, use the pooled SD
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    cohens_d = abs(mean_diff) / pooled_sd
    
    # Round results to 4 decimal places for consistency with example
    return {
        't_statistic': round(t_statistic, 4),
        'p_value': round(p_value, 6),
        'degrees_of_freedom': round(df, 4),
        'reject_null': bool(reject_null),
        'cohens_d': round(cohens_d, 4)
    }