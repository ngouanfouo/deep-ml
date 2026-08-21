import torch

def phi_transform(data: list[float], degree: int) -> torch.Tensor:
    """
    Perform a Phi Transformation to map input features into a higher-dimensional space by generating polynomial features.

    Args:
        data (list[float]): A list of numerical values to transform.
        degree (int): The degree of the polynomial expansion.

    Returns:
        torch.Tensor: A 2D tensor where each row represents the transformed features of a data point,
                      containing powers from 0 to degree.
    """
    if degree < 0:
        return torch.tensor([])
    
    # Convert data to a tensor if it's not already
    data_tensor = torch.tensor(data, dtype=torch.float32)
    
    # If data is 1D, reshape to column vector
    if data_tensor.dim() == 1:
        data_tensor = data_tensor.unsqueeze(1)
    
    # Generate powers from 0 to degree
    result = torch.zeros(len(data), degree + 1)
    for i in range(degree + 1):
        result[:, i] = data_tensor[:, 0] ** i
    
    return result