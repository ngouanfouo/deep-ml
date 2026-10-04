import numpy as np

def backgammon_features(board: list, bar: tuple, borne_off: tuple, turn: str) -> np.ndarray:
    """
    Extract a 198-dimensional feature vector from a backgammon board position.
    
    Args:
        board: list of 24 ints (positive=white, negative=black)
        bar: (white_bar, black_bar)
        borne_off: (white_off, black_off)
        turn: 'white' or 'black'
    
    Returns:
        np.ndarray of shape (198,) with float features
    """
    features = np.zeros(198, dtype=float)
    
    for i in range(24):
        val = board[i]
        
        # White checkers on this point
        n_white = val if val > 0 else 0
        base_white = i * 4
        if n_white >= 1:
            features[base_white] = 1.0
        if n_white >= 2:
            features[base_white + 1] = 1.0
        if n_white >= 3:
            features[base_white + 2] = 1.0
            features[base_white + 3] = (n_white - 3) / 2.0
        
        # Black checkers on this point
        n_black = -val if val < 0 else 0
        base_black = 96 + i * 4
        if n_black >= 1:
            features[base_black] = 1.0
        if n_black >= 2:
            features[base_black + 1] = 1.0
        if n_black >= 3:
            features[base_black + 2] = 1.0
            features[base_black + 3] = (n_black - 3) / 2.0
    
    features[192] = bar[0] / 2.0
    features[193] = bar[1] / 2.0
    features[194] = borne_off[0] / 15.0
    features[195] = borne_off[1] / 15.0
    features[196] = 1.0 if turn == 'white' else 0.0
    features[197] = 1.0 if turn == 'black' else 0.0
    
    return features