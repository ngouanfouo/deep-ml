def nature_cnn_info(input_shape: tuple, num_actions: int) -> dict:
    """
    Compute architecture details for the NatureCNN used in deep Q-networks.

    Args:
        input_shape: tuple of (channels, height, width)
        num_actions: number of output actions

    Returns:
        dict with architecture details
    """
    C_in, H, W = input_shape

    # Conv layer specs: (out_channels, kernel, stride)
    conv_specs = [
        (32, 8, 4),
        (64, 4, 2),
        (64, 3, 1),
    ]

    def conv_out(size, kernel, stride):
        return (size - kernel) // stride + 1

    # --- Track shapes and parameter counts ---
    channels = C_in
    h, w = H, W
    conv_shapes = []
    total_params = 0

    for out_ch, k, s in conv_specs:
        h = conv_out(h, k, s)
        w = conv_out(w, k, s)

        # Weight + bias: k*k*C_in*C_out + C_out
        params = k * k * channels * out_ch + out_ch
        total_params += params

        conv_shapes.append((out_ch, h, w))
        channels = out_ch

    # --- Flatten ---
    flatten_size = channels * h * w

    # --- Fully connected (512 units) ---
    fc_output_size = 512
    total_params += flatten_size * fc_output_size + fc_output_size

    # --- Output layer (linear) ---
    output_size = num_actions
    total_params += fc_output_size * output_size + output_size

    return {
        "conv1_output_shape": conv_shapes[0],
        "conv2_output_shape": conv_shapes[1],
        "conv3_output_shape": conv_shapes[2],
        "flatten_size": flatten_size,
        "fc_output_size": fc_output_size,
        "output_size": output_size,
        "total_params": total_params,
    }