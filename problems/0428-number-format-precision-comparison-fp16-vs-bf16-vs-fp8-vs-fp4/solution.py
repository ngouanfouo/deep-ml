import torch
import math

def quantize_custom(values, E, M, B, has_inf=True, max_val_override=None):
    """
    Quantize values to a custom float format with E exponent bits, 
    M mantissa bits, and bias B using round-to-nearest-even.
    """
    if not isinstance(values, torch.Tensor):
        x = torch.tensor(values, dtype=torch.float64)
    else:
        x = values.clone().detach().to(torch.float64)
        
    shape = x.shape
    x_flat = x.flatten()
    
    # Extract sign
    sign = torch.sign(x_flat)
    sign = torch.where(sign == 0, torch.ones_like(sign), sign)
    abs_x = torch.abs(x_flat)
    
    quantized = torch.zeros_like(x_flat)
    
    # Define maximum representable value
    if max_val_override is not None:
        max_val = max_val_override
    else:
        if has_inf:
            max_biased_exponent = (2 ** E) - 2
            max_m = (2 ** M) - 1
        else:
            max_biased_exponent = (2 ** E) - 1
            max_m = (2 ** M) - 2
            
        max_val = (2.0 ** (max_biased_exponent - B)) * (1.0 + max_m / (2.0 ** M))
        
    nonzero_mask = abs_x > 0
    if torch.any(nonzero_mask):
        a = abs_x[nonzero_mask]
        
        # Calculate exponents
        e = torch.floor(torch.log2(a))
        e_biased = e + B
        
        subnormal_mask = e_biased < 1
        normal_mask = ~subnormal_mask
        
        q_val = torch.zeros_like(a)
        
        # 1. Subnormals
        if torch.any(subnormal_mask):
            m_scale = (2.0 ** M) / (2.0 ** (1 - B))
            m = a[subnormal_mask] * m_scale
            m_rounded = torch.round(m)
            q_val[subnormal_mask] = m_rounded / m_scale
            
        # 2. Normals
        if torch.any(normal_mask):
            e_norm = e[normal_mask]
            f = (a[normal_mask] / (2.0 ** e_norm) - 1.0) * (2.0 ** M)
            f_rounded = torch.round(f)
            q_val[normal_mask] = (2.0 ** e_norm) * (1.0 + f_rounded / (2.0 ** M))
            
        # 3. Handle saturation and overflow limits
        overflow_mask = q_val > max_val
        if torch.any(overflow_mask):
            if has_inf:
                q_val[overflow_mask] = float('inf')
            else:
                q_val[overflow_mask] = max_val
                
        quantized[nonzero_mask] = q_val * sign[nonzero_mask]
        
    return quantized.reshape(shape)

def compare_formats(values) -> dict:
    if isinstance(values, torch.Tensor):
        val_tensor = values.clone().detach().to(torch.float64)
    else:
        val_tensor = torch.tensor(values, dtype=torch.float64)
        
    val_flat = val_tensor.flatten().tolist()
    
    formats_spec = {
        'fp16': {
            'E': 5, 'M': 10, 'B': 15, 'has_inf': True,
            'max_representable': 65504.0,
            'min_positive_normal': 2.0 ** (1 - 15)
        },
        'bf16': {
            'E': 8, 'M': 7, 'B': 127, 'has_inf': True,
            'max_representable': 3.3895313892515355e+38,
            'min_positive_normal': 2.0 ** (1 - 127)
        },
        'fp8_e4m3': {
            'E': 4, 'M': 3, 'B': 7, 'has_inf': False,
            'max_representable': 448.0,
            'min_positive_normal': 0.015625
        },
        'fp4_e2m1': {
            'E': 2, 'M': 1, 'B': 1, 'has_inf': False,
            'max_representable': 4.0,
            'min_positive_normal': 1.0
        }
    }
    
    results = {}
    
    for fmt_name, spec in formats_spec.items():
        max_val_override = spec['max_representable'] if fmt_name == 'fp4_e2m1' else None
        q_tensor = quantize_custom(val_tensor, spec['E'], spec['M'], spec['B'], spec['has_inf'], max_val_override)
        q_list = q_tensor.flatten().tolist()
        
        abs_errors = []
        for orig, quant in zip(val_flat, q_list):
            if math.isinf(quant):
                abs_errors.append(float('inf'))
            else:
                abs_errors.append(abs(orig - quant))
                
        has_inf_error = any(math.isinf(e) for e in abs_errors)
        max_abs_error = float('inf') if has_inf_error else max(abs_errors) if abs_errors else 0.0
        mean_abs_error = float('inf') if has_inf_error else sum(abs_errors) / len(abs_errors) if abs_errors else 0.0
        
        # Helper to format error metrics strictly to 6 decimal places
        def round_err(v):
            if math.isinf(v) or math.isnan(v):
                return v
            return round(float(v), 6)
            
        results[fmt_name] = {
            'max_representable': float(spec['max_representable']),
            'min_positive_normal': float(spec['min_positive_normal']),
            'quantized': [float(v) for v in q_list],
            'max_abs_error': round_err(max_abs_error),
            'mean_abs_error': round_err(mean_abs_error)
        }
        
    return results