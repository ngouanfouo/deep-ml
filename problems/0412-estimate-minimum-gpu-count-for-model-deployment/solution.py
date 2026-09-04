import math
import torch

def estimate_min_gpus(
    num_params_billion: torch.Tensor,
    bytes_per_param: torch.Tensor,
    gpu_memory_gb: torch.Tensor,
    overhead_fraction: torch.Tensor
) -> dict:
    """
    Estimate the minimum number of GPUs needed to deploy a model.

    Args:
        num_params_billion: Number of model parameters in billions (as tensor)
        bytes_per_param: Bytes per parameter (4=FP32, 2=FP16, 1=INT8) (as tensor)
        gpu_memory_gb: Available memory per GPU in GB (as tensor)
        overhead_fraction: Fraction of model memory for runtime overhead (as tensor)

    Returns:
        dict with 'model_memory_gb', 'total_memory_gb', 'min_gpus'
    """
    # Convert to Python scalars if they are tensors
    n = num_params_billion.item() if torch.is_tensor(num_params_billion) else num_params_billion
    bpp = bytes_per_param.item() if torch.is_tensor(bytes_per_param) else bytes_per_param
    gpu_mem = gpu_memory_gb.item() if torch.is_tensor(gpu_memory_gb) else gpu_memory_gb
    ov = overhead_fraction.item() if torch.is_tensor(overhead_fraction) else overhead_fraction

    model_mem = n * bpp
    total_mem = model_mem * (1.0 + ov)
    min_gpus = math.ceil(total_mem / gpu_mem)

    return {
        'model_memory_gb': round(model_mem, 2),
        'total_memory_gb': round(total_mem, 2),
        'min_gpus': min_gpus
    }