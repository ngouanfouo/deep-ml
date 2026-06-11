import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):
	# Your code here
	mean=np.mean(data,axis=0)
	std=np.std(data,axis=0)
	std=np.where(std==0, 1e-8,std)
	standardized_data=(data-mean)/std
	standardized_data=np.round(standardized_data,4)

	min_vals=np.min(data,axis=0)
	max_vals=np.max(data,axis=0)

	range_vals=max_vals-min_vals
	range_vals=np.where(range_vals==0,1,range_vals)
	normalized_data=(data-min_vals)/range_vals
	normalized_data=np.round(normalized_data,4)
	return standardized_data, normalized_data