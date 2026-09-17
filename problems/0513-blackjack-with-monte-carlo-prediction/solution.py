import numpy as np
from collections import defaultdict

def card_value(card):
    """Convert a raw card (1..13) to its Blackjack value (1..10)."""
    if card >= 10:
        return 10
    return card  # ace = 1, 2..9 unchanged

def hand_value(cards):
    """
    Compute the best Blackjack value of a hand and whether it has a usable ace.
    A usable ace is an ace counted as 11 (only possible if it doesn't bust).
    """
    total = 0
    aces = 0
    for c in cards:
        if c == 1:
            aces += 1
            total += 1
        elif c >= 10:
            total += 10
        else:
            total += c
    usable_ace = False
    if aces > 0 and total + 10 <= 21:
        total += 10
        usable_ace = True
    return total, usable_ace

def blackjack_mc_prediction(policy_threshold, num_episodes, gamma=1.0, seed=42):
    """
    Estimate state values for simplified Blackjack using first-visit MC prediction.
    
    Args:
        policy_threshold (int): Player sticks when hand sum >= this value
        num_episodes (int): Number of episodes to simulate
        gamma (float): Discount factor (0 <= gamma <= 1)
        seed (int): Random seed for reproducibility
    
    Returns:
        dict: Mapping of (player_sum, dealer_showing, usable_ace) -> estimated value
    """
    np.random.seed(seed)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)

    for _ in range(num_episodes):
        # Draw initial cards
        player_cards = [np.random.randint(1, 14) for _ in range(2)]
        dealer_cards = [np.random.randint(1, 14) for _ in range(2)]
        dealer_showing = card_value(dealer_cards[0])

        # Player's turn: hit while sum < threshold, record states visited
        player_sum, usable_ace = hand_value(player_cards)
        player_states = []
        while True:
            if player_sum > 21:
                break
            player_states.append((player_sum, dealer_showing, usable_ace))
            if player_sum >= policy_threshold:
                break  # stick
            # hit
            player_cards.append(np.random.randint(1, 14))
            player_sum, usable_ace = hand_value(player_cards)

        # Determine final reward
        if player_sum > 21:
            reward = -1.0
        else:
            # Dealer's turn: hit until 17 or more
            dealer_sum, _ = hand_value(dealer_cards)
            while dealer_sum < 17:
                dealer_cards.append(np.random.randint(1, 14))
                dealer_sum, _ = hand_value(dealer_cards)
            if dealer_sum > 21:
                reward = 1.0
            elif player_sum > dealer_sum:
                reward = 1.0
            elif player_sum < dealer_sum:
                reward = -1.0
            else:
                reward = 0.0

        # First-visit Monte Carlo: process states in reverse order
        T = len(player_states)
        visited_in_episode = set()
        for i in range(T - 1, -1, -1):
            # Return from state i is reward discounted by remaining steps
            G = reward * (gamma ** (T - 1 - i))
            state = player_states[i]
            if state not in visited_in_episode:
                visited_in_episode.add(state)
                returns_sum[state] += G
                returns_count[state] += 1

    # Compute average returns
    V = {}
    for state in returns_sum:
        V[state] = returns_sum[state] / returns_count[state]
    return V