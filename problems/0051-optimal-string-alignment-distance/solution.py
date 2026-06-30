def OSA(source: str, target: str) -> int:
    m, n = len(source), len(target)

    # Initialize a 2D DP table with zeros
    # dp[i][j] will store the OSA distance between source[0...i-1] and target[0...j-1]
    dp = [[0] * (n + 1) for _ in range(m + 1)]

    # Base cases: transforming a string to/from an empty string
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    # Fill the DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            # Cost of substitution (0 if characters match, 1 if they don't)
            cost = 0 if source[i - 1] == target[j - 1] else 1

            dp[i][j] = min(
                dp[i - 1][j] + 1,  # Deletion
                dp[i][j - 1] + 1,  # Insertion
                dp[i - 1][j - 1] + cost,  # Substitution
            )

            # Check for an adjacent Transposition
            if (
                i > 1
                and j > 1
                and source[i - 1] == target[j - 2]
                and source[i - 2] == target[j - 1]
            ):
                dp[i][j] = min(dp[i][j], dp[i - 2][j - 2] + 1)

    return dp[m][n]