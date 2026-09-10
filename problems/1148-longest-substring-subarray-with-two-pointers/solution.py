def longest_unique_substring(s):
    last_seen = {}   # character -> most recent index
    left = 0
    max_len = 0
    for right, ch in enumerate(s):
        # If ch is inside the current window, shrink from the left
        if ch in last_seen and last_seen[ch] >= left:
            left = last_seen[ch] + 1
        last_seen[ch] = right
        max_len = max(max_len, right - left + 1)
    return max_len