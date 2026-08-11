import numpy as np

class SparseDistributedMemory:
    def __init__(self, address_length, word_length, num_hard_locations, activation_radius, seed=42):
        # Initialize random number generator
        self.rng = np.random.RandomState(seed)
        
        # Store parameters
        self.address_length = address_length
        self.word_length = word_length
        self.num_hard_locations = num_hard_locations
        self.activation_radius = activation_radius
        
        # Generate random hard location address vectors (binary)
        # Shape: (num_hard_locations, address_length)
        self.hard_locations = self.rng.randint(0, 2, size=(num_hard_locations, address_length))
        
        # Initialize counter matrix to zeros
        # Shape: (num_hard_locations, word_length)
        self.counters = np.zeros((num_hard_locations, word_length), dtype=np.int64)
    
    def _compute_hamming_distance(self, addr1, addr2):
        """Compute Hamming distance between two binary vectors."""
        return np.sum(addr1 != addr2)
    
    def _activate_locations(self, address):
        """Find all hard locations within activation radius of the address."""
        # Convert address to numpy array if it's a list
        address = np.array(address)
        
        # Compute Hamming distance from address to each hard location
        distances = np.sum(self.hard_locations != address, axis=1)
        
        # Find locations within activation radius
        activated_indices = np.where(distances <= self.activation_radius)[0]
        
        return activated_indices
    
    def write(self, address, word):
        """
        Write word to all activated locations, return number activated.
        
        For each activated location:
        - Increment counter by +1 for bits that are 1 in word
        - Decrement counter by -1 for bits that are 0 in word
        """
        # Convert to numpy arrays
        address = np.array(address)
        word = np.array(word)
        
        # Find activated locations
        activated_indices = self._activate_locations(address)
        num_activated = len(activated_indices)
        
        if num_activated == 0:
            return 0
        
        # Update counters for activated locations
        # For word bits that are 1: add +1
        # For word bits that are 0: add -1 (i.e., subtract 1)
        # We can do this efficiently: for each activated location,
        # add (2*word - 1) to the counters (maps 1->+1, 0->-1)
        update = 2 * word - 1  # Convert 0->-1, 1->+1
        
        # Update all activated locations at once
        self.counters[activated_indices] += update
        
        return num_activated
    
    def read(self, address):
        """
        Read from all activated locations, return thresholded binary list.
        
        Sum counters of activated locations element-wise, then threshold:
        - Sum >= 0 -> 1
        - Sum < 0 -> 0
        If no locations activated, return zeros of length word_length.
        """
        # Convert address to numpy array
        address = np.array(address)
        
        # Find activated locations
        activated_indices = self._activate_locations(address)
        num_activated = len(activated_indices)
        
        if num_activated == 0:
            return [0] * self.word_length
        
        # Sum counters of activated locations
        summed_counters = np.sum(self.counters[activated_indices], axis=0)
        
        # Threshold: sum >= 0 -> 1, sum < 0 -> 0
        result = (summed_counters >= 0).astype(int)
        
        return result.tolist()