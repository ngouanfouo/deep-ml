import math

def normal_pdf(x, mean, std_dev):
    """
    Calculate the probability density function (PDF) of the normal distribution.
    :param x: The value at which the PDF is evaluated.
    :param mean: The mean (μ) of the distribution.
    :param std_dev: The standard deviation (σ) of the distribution.
    """
    # Formula: f(x) = (1 / (σ * sqrt(2π))) * exp(-(x - μ)² / (2σ²))
    
    # Calculate the exponent part
    exponent = -((x - mean) ** 2) / (2 * (std_dev ** 2))
    
    # Calculate the normalization factor
    normalization = 1 / (std_dev * math.sqrt(2 * math.pi))
    
    # Calculate the PDF value
    val = normalization * math.exp(exponent)
    
    # Round to 5 decimal places
    return round(val, 5)