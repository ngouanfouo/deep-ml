import numpy as np


def deterministic_hash(s):
    """Converts a string to a deterministic integer."""
    h = 0
    for c in str(s):
        h = (h * 31 + ord(c)) % (2**31)
    return h


def create_hv(dim, seed):
    """Creates a bipolar hypervector of given dimension using the seed."""
    np.random.seed(seed % (2**32 - 1))
    return np.random.choice([-1, 1], dim)


def create_row_hv(row, dim, random_seeds):
    """Create composite hypervector for a dataset row.

    Hint: For each feature, the value seed should combine the base seed
    with the hashed value using modular arithmetic.
    """
    # Array to accumulate bound hypervectors across all features
    bundled_hv = np.zeros(dim, dtype=np.int32)

    for feature_name, feature_value in row.items():
        # 1. Extract the base feature seed
        base_seed = random_seeds[feature_name]

        # 2. Create the bipolar hypervector for the feature name
        name_hv = create_hv(dim, base_seed)

        # 3. Create the value seed by combining base_seed and hashed value
        val_hash = deterministic_hash(feature_value)
        value_seed = base_seed + val_hash

        # 4. Create the bipolar hypervector for the feature value
        value_hv = create_hv(dim, value_seed)

        # 5. Bind name and value via element-wise multiplication
        bound_hv = name_hv * value_hv

        # 6. Accumulate into the bundling sum
        bundled_hv += bound_hv

    # 7. Normalize the bundled sum back into a bipolar hypervector (-1 or 1)
    # Values >= 0 become 1, values < 0 become -1
    normalized_hv = np.where(bundled_hv >= 0, 1, -1)

    return normalized_hv