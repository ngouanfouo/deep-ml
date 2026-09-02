import numpy as np
import math

def llama3_rope_rescale(inv_freq, original_context_length, low_freq_factor, high_freq_factor, scaling_factor):
    """
    Apply Llama 3.1 style piecewise rescaling to a set of RoPE inverse frequencies.

    Args:
        inv_freq: list or array of base inverse frequencies
        original_context_length: int, the original context length used during pretraining
        low_freq_factor: float, threshold factor defining the low-frequency cutoff
        high_freq_factor: float, threshold factor defining the high-frequency cutoff
        scaling_factor: float, the factor by which low-frequency components are divided

    Returns:
        list of floats: adjusted inverse frequencies, each rounded to 6 decimals
    """
    inv_freq_arr = np.array(inv_freq, dtype=float)
    
    # Compute threshold wavelengths
    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor
    
    # Compute wavelength for each inverse frequency
    wavelen = 2 * math.pi / inv_freq_arr
    
    new_inv_freq = np.empty_like(inv_freq_arr)
    
    # Define frequency band masks
    low_mask = wavelen > low_freq_wavelen
    high_mask = wavelen < high_freq_wavelen
    med_mask = ~(low_mask | high_mask)
    
    # 1. Low-frequency band: stretch (divide by scaling factor)
    new_inv_freq[low_mask] = inv_freq_arr[low_mask] / scaling_factor
    
    # 2. High-frequency band: preserve unchanged
    new_inv_freq[high_mask] = inv_freq_arr[high_mask]
    
    # 3. Medium band: smooth interpolation
    if np.any(med_mask):
        w_med = wavelen[med_mask]
        inv_med = inv_freq_arr[med_mask]
        smooth = (original_context_length / w_med - low_freq_factor) / (high_freq_factor - low_freq_factor)
        new_inv_freq[med_mask] = (1 - smooth) * (inv_med / scaling_factor) + smooth * inv_med
        
    return [round(float(x), 6) for x in new_inv_freq]