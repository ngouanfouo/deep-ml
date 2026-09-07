from collections import defaultdict, deque


def vertical_order(values):
    """
    Returns the vertical order traversal of a binary tree given as a level-order list.
    """
    if not values or values[0] is None:
        return []

    column_table = defaultdict(list)
    queue = deque([(0, 0)])  # (node_index, column)
    
    min_col = 0
    max_col = 0

    while queue:
        node_idx, col = queue.popleft()
        if node_idx < len(values) and values[node_idx] is not None:
            column_table[col].append(values[node_idx])
            min_col = min(min_col, col)
            max_col = max(max_col, col)

            left_idx = 2 * node_idx + 1
            right_idx = 2 * node_idx + 2

            queue.append((left_idx, col - 1))
            queue.append((right_idx, col + 1))

    return [column_table[col] for col in range(min_col, max_col + 1)]