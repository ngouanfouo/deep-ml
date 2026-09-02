import numpy as np

def llama3_rope_inv_freq(dim, base=500000.0, factor=8.0, low_freq_factor=1.0, high_freq_factor=4.0, original_context_length=8192):
    """
    Compute RoPE inverse frequencies with Llama 3.1-style frequency scaling.

    Args:
        dim: head dimension (even integer)
        base: RoPE base (theta)
        factor: scale factor applied to low-frequency components
        low_freq_factor: defines low-frequency wavelength threshold
        high_freq_factor: defines high-frequency wavelength threshold
        original_context_length: pre-training context length

    Returns:
        list of inverse frequencies (length dim/2), each rounded to 8 decimals
    """
    # 1. Compute base inverse frequencies: inv_freq[i] = 1 / (base ** (2*i / dim))
    i_indices = np.arange(dim // 2, dtype=float)
    inv_freq = 1.0 / (base ** (2 * i_indices / dim))
    
    # 2. Compute wavelength for each component: wl = 2 * pi / inv_freq
    wl = 2 * np.pi / inv_freq
    
    # 3. Define thresholds
    low_freq_wavelen = original_context_length / low_freq_factor
    high_freq_wavelen = original_context_length / high_freq_factor
    
    # 4. Apply piecewise scaling rule
    scaled_inv_freq = np.empty_like(inv_freq)
    
    # High frequency: wl < high_freq_wavelen -> keep unchanged
    high_mask = wl < high_freq_wavelen
    scaled_inv_freq[high_mask] = inv_freq[high_mask]
    
    # Low frequency: wl > low_freq_wavelen -> divide by factor
    low_mask = wl > low_freq_wavelen
    scaled_inv_freq[low_mask] = inv_freq[low_mask] / factor
    
    # Medium frequency: otherwise -> linear interpolation using smoothing weight s
    med_mask = ~(high_mask | low_mask)
    if np.any(med_mask):
        wl_med = wl[med_mask]
        inv_med = inv_freq[med_mask]
        s = (original_context_length / wl_med - low_freq_factor) / (high_freq_factor - low_freq_factor)
        scaled_inv_freq[med_mask] = (1 - s) * (inv_med / factor) + s * inv_med
        
    # Return as a list of floats rounded to 8 decimal places
    return [round(float(x), 8) for x in scaled_inv_freq]