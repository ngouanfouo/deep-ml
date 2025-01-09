def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
	if mode=='column':
		return [sum(col)/len(matrix) for col in zip(*matrix)]
	elif mode=='row':
		return [sum(row)/len(row) for row in matrix]