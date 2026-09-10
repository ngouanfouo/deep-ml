def analyze(nested, tree, low, high):
    # --- Part 1 & 2: nested list ---
    def find_max_depth(node, depth):
        max_d = depth
        for item in node:
            if isinstance(item, list):
                max_d = max(max_d, find_max_depth(item, depth + 1))
        return max_d
    
    def top_down(node, depth):
        total = 0
        for item in node:
            if isinstance(item, list):
                total += top_down(item, depth + 1)
            else:
                total += item * depth
        return total
    
    def inverse(node, depth, D):
        total = 0
        for item in node:
            if isinstance(item, list):
                total += inverse(item, depth + 1, D)
            else:
                total += item * (D - depth + 1)
        return total
    
    D = find_max_depth(nested, 1)
    top_down_sum = top_down(nested, 1)
    inverse_sum = inverse(nested, 1, D)
    
    # --- Part 3: BST range sum ---
    def range_sum(node):
        if node is None:
            return 0
        val, left, right = node
        total = 0
        if low <= val <= high:
            total += val
        total += range_sum(left)
        total += range_sum(right)
        return total
    
    range_sum_val = range_sum(tree)
    
    return (top_down_sum, inverse_sum, range_sum_val)