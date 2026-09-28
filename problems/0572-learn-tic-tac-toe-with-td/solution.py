def td_tic_tac_toe(games: list, alpha: float = 0.1, initial_value: float = 0.5) -> dict:
    """
    Learn Tic-Tac-Toe state values using TD(0) from recorded games.

    Args:
        games: List of game dicts, each with 'states' (list of board tuples)
               and 'result' (float: 1.0=win, 0.0=loss, 0.5=draw)
        alpha: Learning rate for TD updates
        initial_value: Initial value for unseen non-terminal states

    Returns:
        Dict mapping non-terminal state tuples to learned float values,
        rounded to 4 decimal places.
    """
    V = {}

    for game in games:
        states = game['states']
        result = game['result']

        # Walk forward through the non-terminal states
        for i in range(len(states) - 1):
            s = states[i]
            next_s = states[i + 1]

            # Initialize on first encounter
            if s not in V:
                V[s] = initial_value

            # Target: value of next state if non-terminal, else the game result
            if i + 1 == len(states) - 1:
                # next_s is the terminal state; use the game result
                target = result
            else:
                if next_s not in V:
                    V[next_s] = initial_value
                target = V[next_s]

            # TD(0) update
            V[s] += alpha * (target - V[s])

    # Round for display
    return {state: round(value, 4) for state, value in V.items()}