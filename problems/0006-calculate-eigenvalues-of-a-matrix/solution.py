def calculate_eigenvalues(matrix: list[list[float|int]]) -> list[float]:
	a,b,c,d=matrix[0][0],matrix[0][1],matrix[1][0],matrix[1][1]
	trace=a+d
	determinant=a*d-b*c
	discriminant=trace**2-4*determinant
	# solve for eigenvalues
	lambda_1=(trace+discriminant**0.5)/2
	lambda_2=(trace-discriminant**0.5)/2
	return [lambda_1,lambda_2]