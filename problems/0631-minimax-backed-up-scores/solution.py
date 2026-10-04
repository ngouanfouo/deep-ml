def minimax_score(tree, is_maximizing=True):
    """
    Compute the minimax backed-up score for a game tree.
    
    Args:
        tree: Nested list game tree (numbers are leaves, lists are internal nodes)
        is_maximizing: True if root player is maximizing
    
    Returns:
        Tuple of (value, optimal_path)
        - value: float, minimax score rounded to 4 decimal places
        - optimal_path: list of int, child indices along optimal play
    """
    def helper(node, maximizing):
        # Leaf node: return its value and an empty path
        if not isinstance(node, list):
            return float(node), []
        
        best_value = None
        best_path = None
        
        for i, child in enumerate(node):
            child_value, child_path = helper(child, not maximizing)
            
            if best_value is None:
                best_value = child_value
                best_path = [i] + child_path
            elif maximizing:
                if child_value > best_value:  # strict > keeps lowest index on ties
                    best_value = child_value
                    best_path = [i] + child_path
            else:
                if child_value < best_value:  # strict < keeps lowest index on ties
                    best_value = child_value
                    best_path = [i] + child_path
        
        return best_value, best_path
    
    value, path = helper(tree, is_maximizing)
    return round(float(value), 4), path