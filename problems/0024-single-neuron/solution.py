import math

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
    # Calculate sigmoid for each feature vector
    probabilities = []
    for feature_vector in features:
        # Compute weighted sum: dot product of features and weights + bias
        weighted_sum = sum(f * w for f, w in zip(feature_vector, weights)) + bias
        # Apply sigmoid activation
        sigmoid = 1 / (1 + math.exp(-weighted_sum))
        probabilities.append(sigmoid)
    
    # Calculate mean squared error
    mse = sum((prob - label) ** 2 for prob, label in zip(probabilities, labels)) / len(labels)
    
    # Round to 4 decimal places
    probabilities = [round(p, 4) for p in probabilities]
    mse = round(mse, 4)
    
    return probabilities, mse