def max_window_sum(nums, k):
    # nums: list of integers
    # k: window length (int)
    # return the maximum sum over all contiguous windows of length k
    n = len(nums)
    if n == 0 or k > n or k <= 0:
        return None
    
    # Sum of the first window
    window_sum = sum(nums[:k])
    max_sum = window_sum
    
    # Slide the window one element at a time
    for i in range(k, n):
        window_sum += nums[i] - nums[i - k]
        if window_sum > max_sum:
            max_sum = window_sum
    
    return max_sum