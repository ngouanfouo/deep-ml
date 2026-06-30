import torch
from typing import Optional
import math

class MCTSNode:
    def __init__(self, state: int, parent: Optional['MCTSNode'] = None):
        self.state = state
        self.parent = parent
        self.children = {}
        self.visits = 0
        self.value = 0.0
    
    def is_leaf(self) -> bool:
        return len(self.children) == 0
    
    def is_terminal(self, max_value: int) -> bool:
        """Check if this state is terminal (game over)."""
        return self.state >= max_value
    
    def ucb1(self, c: float = 1.414) -> float:
        """Upper Confidence Bound for Trees."""
        if self.visits == 0:
            return float('inf')
        if self.parent is None:
            return self.value / self.visits
        exploitation = self.value / self.visits
        exploration = c * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploitation + exploration


def mcts_search(initial_state: int, max_value: int, iterations: int, seed: int = 42) -> int:
    """
    Monte Carlo Tree Search for sequential game.
    
    Args:
        initial_state: Starting value
        max_value: Target value to reach
        iterations: Number of MCTS iterations
        seed: Random seed
    
    Returns:
        Best action (1 or 2)
    """
    # Set random seed for reproducibility
    torch.manual_seed(seed)
    
    # Create root node
    root = MCTSNode(initial_state)
    
    # MCTS iterations
    for _ in range(iterations):
        # Phase 1: Selection
        node = root
        path = [node]
        
        # Select until we reach a leaf or terminal state
        while not node.is_leaf() and not node.is_terminal(max_value):
            # Choose child with highest UCB1 value
            children = list(node.children.values())
            ucb_values = torch.tensor([child.ucb1() for child in children])
            best_idx = torch.argmax(ucb_values).item()
            node = children[best_idx]
            path.append(node)
        
        # Phase 2: Expansion
        # If node is not terminal, expand it
        if not node.is_terminal(max_value):
            # Get available actions from current state
            available_actions = [a for a in [1, 2] if node.state + a <= max_value]
            
            # Add all children (expand)
            for action in available_actions:
                next_state = node.state + action
                if next_state not in node.children:
                    node.children[next_state] = MCTSNode(next_state, node)
            
            # Select a child for simulation (if any)
            if node.children:
                # Choose the first child or the one with highest UCB1
                children = list(node.children.values())
                ucb_values = torch.tensor([child.ucb1() for child in children])
                best_idx = torch.argmax(ucb_values).item()
                node = children[best_idx]
                path.append(node)
        
        # Phase 3: Simulation (random playout)
        winner = simulate(node.state, max_value)
        
        # Phase 4: Backpropagation
        # Update statistics for all nodes in the path
        for path_node in reversed(path):
            path_node.visits += 1
            # Winner is 1 if the player who made the last move wins
            # We store value from the perspective of the node's player
            if winner == 1:
                path_node.value += 1.0
            else:
                path_node.value += 0.0
    
    # Choose the best action from root
    best_action = 1
    best_win_rate = -float('inf')
    
    for next_state, child in root.children.items():
        win_rate = child.value / child.visits if child.visits > 0 else 0
        if win_rate > best_win_rate:
            best_win_rate = win_rate
            best_action = next_state - root.state
    
    return best_action


def simulate(state: int, max_value: int) -> int:
    """
    Perform a random simulation from the given state.
    Returns: 1 if the player who made the last move wins, -1 if they lose.
    """
    current_state = state
    player = 1  # 1 for current player, -1 for opponent
    
    while current_state < max_value:
        # Get available actions
        actions = [a for a in [1, 2] if current_state + a <= max_value]
        
        if not actions:
            # No valid moves (shouldn't happen in this game)
            break
        
        # Randomly choose an action using torch
        action_idx = torch.randint(0, len(actions), (1,)).item()
        action = actions[action_idx]
        current_state += action
        
        # Check if player won
        if current_state == max_value:
            return player  # Current player wins
        elif current_state > max_value:
            return -player  # Current player loses (went over)
        
        # Switch player
        player = -player
    
    # If we exit the loop without a winner
    return -1