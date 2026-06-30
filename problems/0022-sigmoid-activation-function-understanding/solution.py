import numpy as np


def sigmoid(z):
    """Computes the sigmoid activation function for an input z,

    rounded to four decimal places.
    """
    # Using np.exp handles both scalar values and numpy arrays efficiently
    output = 1 / (1 + np.exp(-z))
    return round(float(output), 4)