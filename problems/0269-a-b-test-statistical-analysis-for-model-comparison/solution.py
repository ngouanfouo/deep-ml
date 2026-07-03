import numpy as np
from scipy import stats
import math

def analyze_ab_test(control_outcomes: list, treatment_outcomes: list, confidence_level: float = 0.95, min_detectable_effect: float = 0.02) -> dict:
    """
    Analyze A/B test results for model comparison with statistical rigor.
    
    Args:
        control_outcomes: List of binary outcomes (0 or 1) for control group
        treatment_outcomes: List of binary outcomes (0 or 1) for treatment group
        confidence_level: Confidence level for statistical tests (default 0.95)
        min_detectable_effect: Minimum absolute effect size considered practically significant
    
    Returns:
        dict with statistical analysis results and recommendation
    """
    # If either list is empty, return empty dict
    if not control_outcomes or not treatment_outcomes:
        return {}
    
    # Convert to numpy arrays for efficient computation
    control = np.array(control_outcomes)
    treatment = np.array(treatment_outcomes)
    
    # Sample sizes
    n_control = len(control)
    n_treatment = len(treatment)
    
    # Success counts and rates
    successes_control = np.sum(control)
    successes_treatment = np.sum(treatment)
    rate_control = successes_control / n_control
    rate_treatment = successes_treatment / n_treatment
    
    # Absolute and relative lift
    absolute_lift = rate_treatment - rate_control
    relative_lift_pct = (absolute_lift / rate_control * 100) if rate_control > 0 else 0.0
    
    # Pooled proportion (used for z-test)
    pooled_prop = (successes_control + successes_treatment) / (n_control + n_treatment)
    
    # Standard error for two-proportion z-test (using pooled variance)
    se_pooled = math.sqrt(pooled_prop * (1 - pooled_prop) * (1/n_control + 1/n_treatment))
    
    # Z-statistic (two-proportion z-test)
    z_statistic = absolute_lift / se_pooled if se_pooled > 0 else 0.0
    
    # P-value (two-tailed)
    p_value = 2 * (1 - stats.norm.cdf(abs(z_statistic)))
    
    # Confidence interval (using unpooled standard error)
    se_unpooled = math.sqrt(rate_control * (1 - rate_control) / n_control + 
                           rate_treatment * (1 - rate_treatment) / n_treatment)
    
    # Z-critical value for confidence level
    z_critical = stats.norm.ppf(1 - (1 - confidence_level) / 2)
    
    ci_lower = absolute_lift - z_critical * se_unpooled
    ci_upper = absolute_lift + z_critical * se_unpooled
    
    # Statistical significance
    alpha = 1 - confidence_level
    statistically_significant = p_value < alpha
    
    # Practical significance
    practically_significant = abs(absolute_lift) >= min_detectable_effect
    
    # Required sample size per group for 80% power (using pooled proportion approximation)
    # For two-proportion test with 80% power and significance level alpha
    z_beta = 0.84  # z for 80% power
    # Simplified sample size calculation
    if absolute_lift != 0 and pooled_prop > 0 and pooled_prop < 1:
        required_n = int(2 * (z_critical + z_beta)**2 * pooled_prop * (1 - pooled_prop) / (absolute_lift**2))
    else:
        required_n = 0
    
    # Recommendation logic
    if statistically_significant and practically_significant and absolute_lift > 0:
        recommendation = 'launch_treatment'
    elif statistically_significant and (absolute_lift < 0 or not practically_significant):
        recommendation = 'keep_control'
    else:
        recommendation = 'continue_testing'
    
    return {
        'control_rate': round(rate_control, 4),
        'treatment_rate': round(rate_treatment, 4),
        'absolute_lift': round(absolute_lift, 4),
        'relative_lift_pct': round(relative_lift_pct, 2),
        'z_statistic': round(z_statistic, 4),
        'p_value': round(p_value, 4),
        'ci_lower': round(ci_lower, 4),
        'ci_upper': round(ci_upper, 4),
        'statistically_significant': statistically_significant,
        'practically_significant': practically_significant,
        'required_sample_size': required_n,
        'recommendation': recommendation
    }