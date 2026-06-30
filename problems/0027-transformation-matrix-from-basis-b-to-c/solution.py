import numpy as np


def transform_basis(
    B: list[list[float]], C: list[list[float]]
) -> list[list[float]]:
    # Treat the input lists directly as matrices without transposing
    mat_B = np.array(B)
    mat_C = np.array(C)

    # Compute P = inverse(C) @ B
    mat_P = np.linalg.inv(mat_C) @ mat_B

    # Round and convert back to a nested list
    return np.round(mat_P, 4).tolist()