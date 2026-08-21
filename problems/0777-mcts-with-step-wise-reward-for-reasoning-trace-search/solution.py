import math

def mcts_reasoning_trace(reward_table, branching, depth, n_simulations, c):
    """
    UCB1-MCTS over a B-ary tree of depth `depth` with a step-wise reward model.

    Args:
        reward_table: dict mapping path tuples to step rewards (missing -> 0.0).
        branching: branching factor B (actions 0..B-1 at every internal node).
        depth: maximum trace depth (leaves live at this depth).
        n_simulations: number of MCTS iterations.
        c: UCB1 exploration constant.

    Returns:
        Best complete trace found (list of length `depth`).
    """
    # Node structure: {path: {'visits': int, 'total': float, 'children': list}}
    # Root is represented by empty tuple ()
    tree = {(): {'visits': 0, 'total': 0.0, 'children': []}}
    
    best_trace = None
    best_score = -float('inf')
    
    def get_step_reward(path):
        """Get reward for a path tuple, default 0.0 if not in reward_table."""
        return reward_table.get(path, 0.0)
    
    def get_trace_score(path):
        """Compute total score of a complete trace."""
        total = 0.0
        for t in range(1, len(path) + 1):
            total += get_step_reward(path[:t])
        return total
    
    def get_ucb1_score(child_path, parent_visits):
        """Compute UCB1 score for a child."""
        child_node = tree[child_path]
        visits = child_node['visits']
        if visits == 0:
            return float('inf')
        q = child_node['total'] / visits
        exploration = c * math.sqrt(math.log(parent_visits) / visits)
        return q + exploration
    
    def select_and_expand():
        """Selection and expansion phase. Returns the path to expand from."""
        path = ()  # Start at root
        
        while True:
            node = tree[path]
            # Check if at depth
            if len(path) == depth:
                return path, False  # Terminal node, no expansion
            
            # Check if node is fully expanded
            num_children = len(node['children'])
            if num_children < branching:
                # Not fully expanded: expand with smallest untried action
                # Find smallest action not yet added
                existing_actions = set(child[-1] for child in node['children'])
                for action in range(branching):
                    if action not in existing_actions:
                        new_path = path + (action,)
                        # Add child node
                        tree[new_path] = {'visits': 0, 'total': 0.0, 'children': []}
                        node['children'].append(new_path)
                        # Sort children by action index
                        node['children'].sort(key=lambda p: p[-1])
                        return new_path, True  # Expanded
            else:
                # Fully expanded: select best child by UCB1
                best_child = None
                best_ucb = -float('inf')
                for child_path in node['children']:
                    ucb = get_ucb1_score(child_path, node['visits'])
                    if ucb > best_ucb:
                        best_ucb = ucb
                        best_child = child_path
                path = best_child
    
    def greedy_rollout(path):
        """Extend path to depth using greedy step-reward rollout."""
        current_path = list(path)
        while len(current_path) < depth:
            # Pick action with largest immediate step reward, tie by smallest action
            best_action = 0
            best_reward = get_step_reward(tuple(current_path + [0]))
            for action in range(1, branching):
                reward = get_step_reward(tuple(current_path + [action]))
                if reward > best_reward:
                    best_reward = reward
                    best_action = action
            current_path.append(best_action)
        return tuple(current_path)
    
    def backpropagate(path, total_reward):
        """Backpropagate total_reward along the selection path."""
        for t in range(len(path) + 1):
            node_path = path[:t]
            node = tree[node_path]
            node['visits'] += 1
            node['total'] += total_reward
    
    # Run MCTS simulations
    for _ in range(n_simulations):
        # Selection and expansion
        expanded_path, was_expanded = select_and_expand()
        
        # Rollout from expanded path
        if len(expanded_path) == depth:
            # Terminal node reached
            trace = expanded_path
        else:
            trace = greedy_rollout(expanded_path)
        
        # Compute total reward of the complete trace
        total_reward = get_trace_score(trace)
        
        # Backpropagate along selection path (not including rollout nodes)
        # The path to backpropagate is the original path before expansion
        # plus the expanded node if expansion happened
        if was_expanded:
            # Backpropagate from root through the expanded node
            backpropagate(expanded_path, total_reward)
        else:
            # Terminal node reached without expansion
            backpropagate(expanded_path, total_reward)
        
        # Track best trace (strictly greater comparison)
        if total_reward > best_score:
            best_score = total_reward
            best_trace = list(trace)
    
    return best_trace