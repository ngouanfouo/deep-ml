def subset_sum_all(nums, target):
    n = len(nums)
    all_non_negative = all(x >= 0 for x in nums)

    suffix_sum = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        suffix_sum[i] = suffix_sum[i + 1] + nums[i]

    result = []

    def dfs(i, current_sum, chosen):
        if all_non_negative:
            if current_sum > target:
                return
            if current_sum + suffix_sum[i] < target:
                return

        if i == n:
            if current_sum == target:
                result.append(chosen.copy())
            return

        # include nums[i]
        chosen.append(nums[i])
        dfs(i + 1, current_sum + nums[i], chosen)
        chosen.pop()

        # exclude nums[i]
        dfs(i + 1, current_sum, chosen)

    dfs(0, 0, [])
    return result


def count_subset_sums(nums, target):
    dp = {0: 1}
    for x in nums:
        new_dp = dict(dp)
        for s, cnt in dp.items():
            new_dp[s + x] = new_dp.get(s + x, 0) + cnt
        dp = new_dp
    return dp.get(target, 0)