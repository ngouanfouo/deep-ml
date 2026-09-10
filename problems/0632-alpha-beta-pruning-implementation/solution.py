def alpha_beta_pruning(tree, is_maximizing=True):
    """
    Perform alpha-beta pruning on a game tree.

    Args:
        tree: A nested list representing the game tree. Leaf nodes are numbers,
              internal nodes are lists of children.
        is_maximizing: Whether the root node is a maximizing player.

    Returns:
        A tuple (value, nodes_evaluated) where value is the optimal minimax
        value and nodes_evaluated is the number of leaf nodes examined.
    """
    NEG_INF = float('-inf')
    POS_INF = float('inf')

    nodes_evaluated = 0

    def search(node, is_max, alpha, beta):
        nonlocal nodes_evaluated

        # Leaf: a single number
        if not isinstance(node, list):
            nodes_evaluated += 1
            return node

        if is_max:
            value = NEG_INF
            for child in node:
                child_val = search(child, False, alpha, beta)
                if child_val > value:
                    value = child_val
                if value > alpha:
                    alpha = value
                if alpha >= beta:
                    break  # beta cutoff
            return value
        else:
            value = POS_INF
            for child in node:
                child_val = search(child, True, alpha, beta)
                if child_val < value:
                    value = child_val
                if value < beta:
                    beta = value
                if alpha >= beta:
                    break  # alpha cutoff
            return value

    root_value = search(tree, is_maximizing, NEG_INF, POS_INF)
    return (root_value, nodes_evaluated)