def array_drills(op, *args):
    if op == "product":
        nums = args[0]
        n = len(nums)
        out = [1] * n
        # Prefix products
        prefix = 1
        for i in range(n):
            out[i] = prefix
            prefix *= nums[i]
        # Suffix products multiplied in
        suffix = 1
        for i in range(n - 1, -1, -1):
            out[i] *= suffix
            suffix *= nums[i]
        return out

    elif op == "3sum":
        nums = args[0]
        nums = sorted(nums)
        n = len(nums)
        result = []
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue  # skip duplicate first element
            if nums[i] > 0:
                break  # no possible triplet sums to 0 after this
            lo, hi = i + 1, n - 1
            while lo < hi:
                s = nums[i] + nums[lo] + nums[hi]
                if s < 0:
                    lo += 1
                elif s > 0:
                    hi -= 1
                else:
                    result.append([nums[i], nums[lo], nums[hi]])
                    # Skip duplicates for lo and hi
                    while lo < hi and nums[lo] == nums[lo + 1]:
                        lo += 1
                    while lo < hi and nums[hi] == nums[hi - 1]:
                        hi -= 1
                    lo += 1
                    hi -= 1
        return result

    elif op == "peak":
        nums = args[0]
        lo, hi = 0, len(nums) - 1
        while lo < hi:
            mid = (lo + hi) // 2
            if nums[mid] < nums[mid + 1]:
                lo = mid + 1
            else:
                hi = mid
        return lo

    elif op == "maxswap":
        num = args[0]
        digits = list(str(num))
        n = len(digits)
        # Track rightmost position of each digit (0-9)
        last = [-1] * 10
        for i, d in enumerate(digits):
            last[int(d)] = i
        # Find the first position where we can swap in a larger digit
        for i in range(n):
            for d in range(9, int(digits[i]), -1):
                if last[d] > i:
                    # Swap
                    j = last[d]
                    digits[i], digits[j] = digits[j], digits[i]
                    return int(''.join(digits))
        return num  # no beneficial swap

    elif op == "merge":
        nums1, m, nums2, n = args
        result = []
        i = j = 0
        while i < m and j < n:
            if nums1[i] <= nums2[j]:
                result.append(nums1[i])
                i += 1
            else:
                result.append(nums2[j])
                j += 1
        while i < m:
            result.append(nums1[i])
            i += 1
        while j < n:
            result.append(nums2[j])
            j += 1
        return result

    else:
        raise ValueError(f"Unknown op: {op}")