def leaky_relu(z: float, alpha: float = 0.01) -> float|int:
	# Your code here
	return  max(alpha*z,z)
