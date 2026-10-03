def compute_receptive_field(input_size: int, layers: list) -> dict:
    """
    Compute the receptive field size and output spatial size
    for a 1D convolutional network.
    """
    receptive_field = 1
    jump = 1          # cumulative stride: distance between two adjacent output centers
    out_size = input_size

    for (kernel_size, stride, padding) in layers:
        # Receptive field grows by the kernel's extra extent scaled by the current jump
        receptive_field += (kernel_size - 1) * jump
        jump *= stride

        # Spatial output size (floor division)
        out_size = (out_size + 2 * padding - kernel_size) // stride + 1

    return {
        "receptive_field": int(receptive_field),
        "output_size": int(out_size),
    }