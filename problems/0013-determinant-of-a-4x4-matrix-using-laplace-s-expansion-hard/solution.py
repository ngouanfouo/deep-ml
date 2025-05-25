def determinant_4x4(matrix: list[list[float]]) -> float:
    """
    Calculates the determinant of a 4x4 matrix using Laplace's Expansion.

    Args:
        matrix: A 4x4 matrix represented as a list of lists.

    Returns:
        The determinant of the matrix.

    Raises:
        ValueError: If the input matrix is not a 4x4 matrix.
    """
    if not (isinstance(matrix, list) and len(matrix) == 4 and
            all(isinstance(row, list) and len(row) == 4 for row in matrix)):
        raise ValueError("Input matrix must be a 4x4 list of lists.")

    def _determinant_3x3(m: list[list[float]]) -> float:
        """Helper to calculate the determinant of a 3x3 matrix."""
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )

    def _get_minor(m: list[list[float]], row: int, col: int) -> list[list[flo