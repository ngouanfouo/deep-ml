def dp_drills(weights, values, capacity, m, n, s1, s2):
    # --- 0/1 Knapsack ---
    # dp[w] = max value with total weight <= w
    dp_k = [0] * (capacity + 1)
    for wi, vi in zip(weights, values):
        for w in range(capacity, wi - 1, -1):
            cand = dp_k[w - wi] + vi
            if cand > dp_k[w]:
                dp_k[w] = cand
    knapsack = dp_k[capacity]
    
    # --- Grid paths ---
    if m <= 0 or n <= 0:
        grid_paths = 0
    else:
        dp_g = [[1] * n for _ in range(m)]
        for i in range(1, m):
            for j in range(1, n):
                dp_g[i][j] = dp_g[i-1][j] + dp_g[i][j-1]
        grid_paths = dp_g[m-1][n-1]
    
    # --- LCS ---
    len1, len2 = len(s1), len(s2)
    dp_l = [[0] * (len2 + 1) for _ in range(len1 + 1)]
    for i in range(1, len1 + 1):
        for j in range(1, len2 + 1):
            if s1[i-1] == s2[j-1]:
                dp_l[i][j] = dp_l[i-1][j-1] + 1
            else:
                dp_l[i][j] = max(dp_l[i-1][j], dp_l[i][j-1])
    lcs = dp_l[len1][len2]
    
    complexity = {
        'knapsack': 'O(n*W) time, O(n*W) space',
        'grid_paths': 'O(m*n) time, O(m*n) space',
        'lcs': 'O(m*n) time, O(m*n) space',
    }
    
    return {
        'knapsack': knapsack,
        'grid_paths': grid_paths,
        'lcs': lcs,
        'complexity': complexity,
    }