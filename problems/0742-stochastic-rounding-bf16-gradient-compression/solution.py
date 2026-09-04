import numpy as np

def stochastic_round_bf16(grads, rand_ints):
    """
    Stochastically round FP32 gradients to BF16 precision.
    
    Args:
        grads: list of Python floats (FP32 gradients)
        rand_ints: list of ints in [0, 65536), one per gradient,
                   representing random draws for stochastic rounding.
    
    Returns:
        list of floats: stochastically rounded values (stored as FP32).
    """
    result = []
    for grad, rand in zip(grads, rand_ints):
        # Get raw 32-bit unsigned representation of the FP32 value
        bits = np.float32(grad).view(np.uint32).item()
        
        # Split into upper 16 bits (BF16 grid point) and lower 16 bits
        upper = bits & 0xFFFF0000
        lower = bits & 0x0000FFFF
        
        # Stochastic rounding:
        # round up (add 1 to the mantissa in the upper 16 bits)
        # with probability lower / 65536.
        # This is equivalent to: if rand >= (65536 - lower) then round up.
        threshold = 0x10000 - lower  # 65536 - lower
        if rand >= threshold:
            rounded_bits = upper + 0x00010000
        else:
            rounded_bits = upper
        
        # Convert back to float (as Python float)
        rounded_val = np.uint32(rounded_bits).view(np.float32).item()
        result.append(rounded_val)
    
    return result