def count_and_longest_subarrays(nums, k):
    # nums: list of ints, k: int
    # return a tuple (count, longest)
    count_map = {0: 1}      # prefix sum -> number of times seen
    first_index = {0: 0}    # prefix sum -> earliest index where it appeared
    
    count = 0
    longest = 0
    prefix = 0
    
    for j, num in enumerate(nums, start=1):
        prefix += num
        target = prefix - k
        
        # Counting: all previous occurrences of `target` give a valid subarray
        if target in count_map:
            count += count_map[target]
            # Longest: use earliest occurrence of `target`
            length = j - first_index[target]
            if length > longest:
                longest = length
        
        # Update maps
        count_map[prefix] = count_map.get(prefix, 0) + 1
        if prefix not in first_index:
            first_index[prefix] = j
    
    return (count, longest)