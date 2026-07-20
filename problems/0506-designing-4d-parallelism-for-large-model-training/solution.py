import numpy as np

def design_4d_parallelism(model_config: dict, parallel_config: dict, hardware_config: dict) -> dict:
    """
    Analyze a 4D parallelism configuration for large model training.
    """
    # Extract model configurations
    L = model_config['num_layers']
    H = model_config['hidden_size']
    V = model_config['vocab_size']
    S = model_config['seq_len']
    B = model_config['micro_batch_size']
    num_micro_batches = model_config['num_micro_batches']

    # Extract parallel configurations
    dp = parallel_config['dp']
    tp = parallel_config['tp']
    pp = parallel_config['pp']
    sp = parallel_config['sp']

    # Extract hardware configurations
    num_gpus = hardware_config['num_gpus']
    memory_per_gpu_gb = hardware_config['memory_per_gpu_gb']

    # 1. Validity check
    valid = bool((dp * tp * pp * sp) == num_gpus)

    # 2. Parameter computation
    total_params = int(2 * V * H + L * 12 * H * H)

    # 3. Memory footprints (in GB)
    param_mem = (total_params * 16) / (tp * pp * 1e9)
    activation_mem = (L * 10 * S * B * H * 2) / (pp * tp * sp * 1e9)
    total_mem = param_mem + activation_mem
    fits_in_memory = bool(total_mem <= memory_per_gpu_gb)

    # 4. Layers per pipeline stage
    layers_per_stage = L / pp

    # Base tensor sizes in bytes (fp16 = 2 bytes)
    sbh_bytes = S * B * H * 2
    grad_bytes_per_rank = (total_params * 2) / (tp * pp)

    # Data Parallelism: Ring all-reduce on local gradients
    if dp > 1:
        dp_comm = 2 * (dp - 1) / dp * grad_bytes_per_rank / 1e9
    else:
        dp_comm = 0.0

    # Tensor Parallelism: 4 ring all-reduces per layer per stage (reported per micro-batch)
    if tp > 1:
        tp_comm = 4 * layers_per_stage * (2 * (tp - 1) / tp * sbh_bytes) / 1e9
    else:
        tp_comm = 0.0

    # Pipeline Parallelism: Total forward + backward boundary volume averaged across the PP chain
    if pp > 1:
        pp_comm = (2 * num_micro_batches * (pp - 1) * sbh_bytes) / (pp * tp * 1e9)
    else:
        pp_comm = 0.0

    # Sequence Parallelism: 2 ring operations per layer per stage (reported per micro-batch)
    if sp > 1:
        sp_comm = 2 * layers_per_stage * ((sp - 1) / sp * sbh_bytes) / 1e9
    else:
        sp_comm = 0.0

    # 5. Pipeline bubble fraction
    bubble_ratio = float((pp - 1) / (num_micro_batches + pp - 1))

    return {
        'valid': valid,
        'total_params': total_params,
        'param_memory_per_gpu_gb': round(param_mem, 4),
        'activation_memory_per_gpu_gb': round(activation_mem, 4),
        'total_memory_per_gpu_gb': round(total_mem, 4),
        'fits_in_memory': fits_in_memory,
        'dp_comm_gb': round(dp_comm, 4),
        'tp_comm_gb': round(tp_comm, 4),
        'pp_comm_gb': round(pp_comm, 4),
        'sp_comm_gb': round(sp_comm, 4),
        'bubble_ratio': round(bubble_ratio, 4)
    }