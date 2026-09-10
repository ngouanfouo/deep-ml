import random

def reservoir_sample(stream, k, seed):
    # Return a list of at most k items sampled uniformly from the stream
    # using a single pass and O(k) memory.
    rng = random.Random(seed)
    reservoir = []
    for i, item in enumerate(stream):
        if i < k:
            reservoir.append(item)
        else:
            j = rng.randint(0, i)   # inclusive on both ends → i+1 possible values
            if j < k:
                reservoir[j] = item
    return reservoir