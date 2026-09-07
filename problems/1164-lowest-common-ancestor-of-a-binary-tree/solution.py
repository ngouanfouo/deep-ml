from collections import deque


class TreeNode:
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None


def lowest_common_ancestor(tree, p, q):
    """
    Returns the value of the lowest common ancestor of nodes p and q
    in a binary tree given as a level-order list.
    """
    if not tree or tree[0] is None:
        return None

    # Step 1: Reconstruct the tree from the level-order list
    root = TreeNode(tree[0])
    queue = deque([root])
    i = 1
    while queue and i < len(tree):
        node = queue.popleft()
        
        if i < len(tree) and tree[i] is not None:
            node.left = TreeNode(tree[i])
            queue.append(node.left)
        i += 1
        
        if i < len(tree) and tree[i] is not None:
            node.right = TreeNode(tree[i])
            queue.append(node.right)
        i += 1

    # Step 2: Find the LCA using standard recursive traversal
    def find_lca(curr, val1, val2):
        if not curr or curr.val == val1 or curr.val == val2:
            return curr
        
        left_res = find_lca(curr.left, val1, val2)
        right_res = find_lca(curr.right, val1, val2)
        
        if left_res and right_res:
            return curr
        return left_res if left_res else right_res

    lca_node = find_lca(root, p, q)
    return lca_node.val if lca_node else None