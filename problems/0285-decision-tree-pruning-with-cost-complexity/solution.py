import torch

def compute_pruning_alphas(tree: dict) -> torch.Tensor:
    """
    Computes effective alpha values for cost-complexity pruning using PyTorch.
    
    Args:
        tree: Dictionary representing a decision tree node with keys:
              - 'samples': number of samples reaching this node
              - 'errors': misclassification count if node becomes a leaf
              - 'left': left child subtree (dict) or None
              - 'right': right child subtree (dict) or None
        
    Returns:
        torch.Tensor of effective alpha values for internal nodes, sorted ascending.
    """
    alphas = []
    
    def traverse(node: dict) -> tuple:
        """
        Traverse the tree and compute alpha values for internal nodes.
        
        Returns:
            Tuple of (leaf_errors, num_leaves) for the subtree rooted at node
        """
        # Check if leaf node (both children are None)
        if node['left'] is None and node['right'] is None:
            # Leaf node: return its error and leaf count 1
            return node['errors'], 1
        
        # Internal node: recursively process children
        left_errors, left_leaves = traverse(node['left'])
        right_errors, right_leaves = traverse(node['right'])
        
        # Compute subtree statistics
        subtree_errors = left_errors + right_errors
        subtree_leaves = left_leaves + right_leaves
        
        # Compute effective alpha for this internal node
        # alpha = (R(t) - R(T_t)) / (|T_t| - 1)
        node_error = node['errors']
        if subtree_leaves > 1:
            alpha = (node_error - subtree_errors) / (subtree_leaves - 1)
            alphas.append(alpha)
        
        # Return aggregated statistics for this subtree
        return subtree_errors, subtree_leaves
    
    # Start traversal from root
    traverse(tree)
    
    # Sort alphas in ascending order and convert to tensor
    alphas.sort()
    return torch.tensor(alphas, dtype=torch.float32)