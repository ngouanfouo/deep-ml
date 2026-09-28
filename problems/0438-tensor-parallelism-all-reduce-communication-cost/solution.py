def tensor_parallel_allreduce_cost(
    hidden_size: int,
    sequence_length: int,
    batch_size: int,
    num_gpus: int,
    num_layers: int,
    bytes_per_element: int = 2,
    bandwidth_gb_per_sec: float = 300.0,
    allreduces_per_layer: int = 2
) -> dict:
    """
    Calculate the all-reduce communication cost for tensor parallelism.

    Args:
        hidden_size: Hidden dimension of the model
        sequence_length: Sequence length
        batch_size: Micro-batch size per GPU
        num_gpus: Number of GPUs in tensor parallel group
        num_layers: Number of transformer layers
        bytes_per_element: Bytes per element (2 for FP16, 4 for FP32)
        bandwidth_gb_per_sec: Interconnect bandwidth in GB/s
        allreduces_per_layer: Number of all-reduce ops per layer (forward pass)

    Returns:
        Dictionary with communication cost analysis
    """
    # 1) Size of the activation tensor being all-reduced.
    #    Shape: (batch_size, sequence_length, hidden_size)
    num_elements = batch_size * sequence_length * hidden_size
    message_size_bytes = int(num_elements * bytes_per_element)

    # 2) Single all-reduce volume per GPU using the ring algorithm.
    #    Ring all-reduce: each GPU sends/receives 2*(n-1)/n * message_size bytes.
    if num_gpus <= 1:
        comm_volume_per_allreduce = 0.0
    else:
        comm_volume_per_allreduce = (
            2.0 * (num_gpus - 1) / num_gpus * message_size_bytes
        )

    # 3) Total volume across all layers and all-reduces per layer.
    total_allreduces = num_layers * allreduces_per_layer
    total_comm_volume_bytes = comm_volume_per_allreduce * total_allreduces

    # 4) Communication time in milliseconds.
    #    bytes / (GB/s * 1e9) * 1e3  →  ms
    if bandwidth_gb_per_sec > 0:
        total_comm_time_ms = (
            total_comm_volume_bytes / (bandwidth_gb_per_sec * 1e9) * 1e3
        )
    else:
        total_comm_time_ms = 0.0

    return {
        'message_size_bytes': message_size_bytes,
        'comm_volume_per_allreduce_bytes': round(comm_volume_per_allreduce, 4),
        'total_comm_volume_bytes': round(total_comm_volume_bytes, 4),
        'total_comm_time_ms': round(total_comm_time_ms, 4),
    }