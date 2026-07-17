import numpy as np

class GradientBandit:
    def __init__(self, num_actions, alpha=0.1):
        """
        num_actions (int): Number of possible actions
        alpha (float): Step size for preference updates
        """
        self.num_actions = num_actions
        self.alpha = alpha
        self.preferences = np.zeros(num_actions)
        self.avg_reward = 0.0
        self.time = 0
    
    def softmax(self):
        # Compute softmax probabilities from preferences
        # Subtract max for numerical stability
        pref_shifted = self.preferences - np.max(self.preferences)
        exp_pref = np.exp(pref_shifted)
        return exp_pref / np.sum(exp_pref)
    
    def select_action(self):
        # Sample an action according to the softmax distribution
        probs = self.softmax()
        return np.random.choice(self.num_actions, p=probs)
    
    def update(self, action, reward):
        # Update action preferences using the gradient ascent update
        self.time += 1
        
        # Update baseline (average reward) with incremental update
        self.avg_reward += (reward - self.avg_reward) / self.time
        
        # Compute softmax probabilities
        probs = self.softmax()
        
        # Create one-hot encoding for the selected action
        one_hot = np.zeros(self.num_actions)
        one_hot[action] = 1.0
        
        # Gradient ascent update for preferences
        # H(a) += alpha * (R - avg_reward) * (1 - pi(a)) for selected action
        # H(a) += alpha * (R - avg_reward) * (0 - pi(a)) for other actions
        reward_baseline = reward - self.avg_reward
        self.preferences += self.alpha * reward_baseline * (one_hot - probs)