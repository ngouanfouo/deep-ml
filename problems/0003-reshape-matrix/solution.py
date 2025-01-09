import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
	array=np.array(a)
	reshaped_matrix=array.reshape(new_shape).tolist()
	return reshaped_matrix