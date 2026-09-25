import math

def backprop_chain(a0: float, y: float, w: list, b: list) -> dict:
    """
    Compute gradients of C = (a_L - y)^2 with respect to each weight and bias
    in a chain network where every layer has a single sigmoid neuron.

    Args:
        a0: input activation (scalar)
        y: target value (scalar)
        w: list of L weights
        b: list of L biases

    Returns:
        Dictionary with keys 'dW' and 'dB', each a list of L floats rounded to 6 places.
    """
    L = len(w)

    # Forward pass: store pre-activations z_l and activations a_l.
    a = [a0]
    z = []

    for l in range(L):
        z_l = w[l] * a[l] + b[l]
        a_l = 1.0 / (1.0 + math.exp(-z_l))
        z.append(z_l)
        a.append(a_l)

    # Backward pass.
    dW = [0.0] * L
    dB = [0.0] * L

    # dC/da_L = 2 * (a_L - y)
    dC_da = 2.0 * (a[L] - y)

    for l in range(L - 1, -1, -1):
        # sigmoid derivative: a_l * (1 - a_l), where a_l = a[l+1]
        sig_deriv = a[l + 1] * (1.0 - a[l + 1])

        # dC/dz_l
        dC_dz = dC_da * sig_deriv

        # Gradients for this layer
        dW[l] = dC_dz * a[l]
        dB[l] = dC_dz

        # Propagate gradient to previous activation
        dC_da = dC_dz * w[l]

    return {
        'dW': [round(x, 6) for x in dW],
        'dB': [round(x, 6) for x in dB],
    }