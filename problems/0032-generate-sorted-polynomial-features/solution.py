import numpy as np
from itertools import combinations_with_replacement,product


def polynomial_features(X, degree):
    # ✏️  Your code here
    n_samples,n_features=X.shape
    exponent_combinations=[]
    for exponents in product(range(degree+1),repeat=n_features):
        if sum(exponents)<=degree:
            exponent_combinations.append(exponents)

    result=[]
    for i in range(n_samples):
        features=[]
        for exponents in exponent_combinations:
            term=1.0
            for j in range(n_features):
                term*=(X[i,j]**exponents[j])
            features.append(term)
        features.sort()
        result.append(features)
    return np.array(result)