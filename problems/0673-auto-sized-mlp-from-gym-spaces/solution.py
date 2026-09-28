import math


def auto_mlp_from_spaces(obs_space: dict, act_space: dict, hidden_layers: list) -> dict:
    """
    Compute MLP architecture details from observation and action space definitions.

    Args:
        obs_space: dict describing the observation space
        act_space: dict describing the action space
        hidden_layers: list of hidden layer widths

    Returns:
        dict with input_dim, output_dim, layer_shapes, total_params
    """

    def space_dim(space: dict) -> int:
        t = space["type"]
        if t == "Box":
            shape = space.get("shape", ())
            return math.prod(shape) if shape else 1
        elif t == "Discrete":
            return int(space["n"])
        elif t == "MultiBinary":
            return int(space["n"])
        else:
            raise ValueError(f"Unsupported space type: {t!r}")

    input_dim = space_dim(obs_space)
    output_dim = space_dim(act_space)

    # Layer dimensions: input -> hidden_layers... -> output
    dims = [input_dim] + list(hidden_layers) + [output_dim]

    layer_shapes = []
    total_params = 0

    for in_features, out_features in zip(dims[:-1], dims[1:]):
        weight_shape = (in_features, out_features)
        bias_shape = (out_features,)
        layer_shapes.append((weight_shape, bias_shape))
        total_params += in_features * out_features + out_features

    return {
        "input_dim": input_dim,
        "output_dim": output_dim,
        "layer_shapes": layer_shapes,
        "total_params": total_params,
    }