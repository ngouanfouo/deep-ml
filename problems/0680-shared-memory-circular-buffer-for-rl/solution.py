import numpy as np


class SharedCircularBuffer:
    def __init__(self, capacity: int, state_dim: int):
        """
        Initialize a circular replay buffer with pre-allocated numpy arrays.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: Dimensionality of state vectors
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.pos = 0
        self.size = 0

        # Pre-allocate fixed-size numpy arrays simulating shared memory segments
        self.states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float64)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float64)
        self.dones = np.zeros(capacity, dtype=bool)

    def push(self, state, action: int, reward: float, next_state, done: bool):
        """Add a transition to the buffer, overwriting oldest if full."""
        self.states[self.pos] = state
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.next_states[self.pos] = next_state
        self.dones[self.pos] = done

        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.capacity, self.size + 1)

    def sample(self, batch_size: int, seed=None) -> dict:
        """Randomly sample a batch of transitions."""
        rng = np.random.RandomState(seed)
        indices = rng.randint(0, self.size, size=batch_size)

        return {
            'states': self.states[indices].tolist(),
            'actions': self.actions[indices].tolist(),
            'rewards': self.rewards[indices].tolist(),
            'next_states': self.next_states[indices].tolist(),
            'dones': self.dones[indices].tolist(),
        }

    def latest(self, n: int) -> dict:
        """Return the n most recent transitions, oldest to newest."""
        k = min(n, self.size)
        indices = (self.pos - k + np.arange(k)) % self.capacity

        return {
            'states': self.states[indices].tolist(),
            'actions': self.actions[indices].tolist(),
            'rewards': self.rewards[indices].tolist(),
            'next_states': self.next_states[indices].tolist(),
            'dones': self.dones[indices].tolist(),
        }

    def __len__(self) -> int:
        """Return current number of stored transitions."""
        return self.size