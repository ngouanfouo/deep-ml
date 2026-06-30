import numpy as np


def translate_object(points, tx, ty):
    """Applies a 2D translation matrix to a list of [x, y] coordinates using

    homogeneous coordinates.
    """
    # 1. Define the 3x3 transformation matrix
    T = np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])

    # 2. Convert input points to a NumPy array
    pts = np.array(points)  # Shape: (N, 2)

    # 3. Convert to homogeneous coordinates by adding a column of ones
    # Shape becomes (N, 3)
    ones = np.ones((pts.shape[0], 1))
    pts_homogeneous = np.hstack((pts, ones))

    # 4. Perform matrix multiplication: T @ points_transpose
    # T is (3, 3) and pts_homogeneous.T is (3, N) -> Result is (3, N)
    translated_homogeneous = np.dot(T, pts_homogeneous.T)

    # 5. Extract x and y coordinates, transpose back to (N, 2), and convert to list
    translated_points = translated_homogeneous[:2, :].T

    return translated_points.tolist()