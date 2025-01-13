def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:

    num_features = len(vectors)
    num_observations = len(vectors[0])

    means = [sum(feature) / num_observations for feature in vectors]

    covariance_matrix = [[0 for _ in range(num_features)] for _ in range(num_features)]

    for i in range(num_features):
        for j in range(i, num_features):
            covariance = 0
            for k in range(num_observations): 
                covariance += (vectors[i][k] - means[i]) * (vectors[j][k] - means[j])
            covariance /= (num_observations - 1)
            covariance_matrix[i][j] = covariance
            covariance_matrix[j][i] = covariance

    return covariance_matrix