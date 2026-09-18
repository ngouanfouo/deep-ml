import numpy as np

class ReplayBuffer:
    """
    A fixed-size circular buffer to store and sample experience tuples
    for off-policy reinforcement learning.
    """
    def __init__(self, capacity: int):
        """Initialize buffer with given maximum capacity."""
        self.capacity = capacity
        self.buffer = []
        self.position = 0  # index of next slot to write (circular)

    def add(self, state, action, reward, next_state, done):
        """Store a transition in the buffer."""
        transition = (
            np.asarray(state, dtype=np.float64),
            action,
            float(reward),
            np.asarray(next_state, dtype=np.float64),
            bool(done),
        )
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.position] = transition
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int, seed: int = None) -> dict:
        """
        Randomly sample a batch of transitions without replacement.
        Returns dict with keys: 'states', 'actions', 'rewards', 'next_states', 'dones'.
        """
        rng = np.random.RandomState(seed)
        n = len(self.buffer)
        if batch_size > n:
            raise ValueError(
                f"Cannot sample {batch_size} transitions from buffer of size {n}."
            )
        indices = rng.choice(n, size=batch_size, replace=False)

        states = np.stack([self.buffer[i][0] for i in indices])
        actions = np.array([self.buffer[i][1] for i in indices])
        rewards = np.array([self.buffer[i][2] for i in indices])
        next_states = np.stack([self.buffer[i][3] for i in indices])
        dones = np.array([self.buffer[i][4] for i in indices])

        return {
            'states': states,
            'actions': actions,
            'rewards': rewards,
            'next_states': next_states,
            'dones': dones,
        }

    def size(self) -> int:
        """Return current number of stored transitions."""
        return len(self.buffer)