import numpy as np

def is_winner(board, player):
    # Check rows, columns, and diagonals
    for i in range(3):
        if all(board[i, j] == player for j in range(3)):
            return True
        if all(board[j, i] == player for j in range(3)):
            return True
    if all(board[i, i] == player for i in range(3)):
        return True
    if all(board[i, 2-i] == player for i in range(3)):
        return True
    return False

def is_full(board):
    return not any(board[i, j] == '' for i in range(3) for j in range(3))

def get_available_moves(board):
    return [(i, j) for i in range(3) for j in range(3) if board[i, j] == '']

def _minimax(board, depth, is_maximizing, ai_player, opponent):
    # Terminal base cases: Score the current board configuration
    if is_winner(board, ai_player):
        return 10 - depth  # Prioritize faster wins
    if is_winner(board, opponent):
        return depth - 10  # Prioritize stalling/defending longer
    if is_full(board):
        return 0

    moves = get_available_moves(board)

    if is_maximizing:
        best_score = -float('inf')
        for r, c in moves:
            board[r, c] = ai_player
            score = _minimax(board, depth + 1, False, ai_player, opponent)
            board[r, c] = ''  # Undo move
            best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for r, c in moves:
            board[r, c] = opponent
            score = _minimax(board, depth + 1, True, ai_player, opponent)
            board[r, c] = ''  # Undo move
            best_score = min(score, best_score)
        return best_score

def minimax_tictactoe(board: np.ndarray, player: str) -> tuple:
    """
    Returns the optimal move (row, col) for the given player ('X' or 'O') 
    on the current board using Minimax.
    """
    opponent = 'O' if player == 'X' else 'X'
    best_score = -float('inf')
    best_move = None

    # Step through all top-level possible actions
    for r, c in get_available_moves(board):
        board[r, c] = player
        # Compute value of this decision pathway
        score = _minimax(board, depth=0, is_maximizing=False, ai_player=player, opponent=opponent)
        board[r, c] = ''  # Backtrack Clean-up

        if score > best_score:
            best_score = score
            best_move = (r, c)

    return best_move