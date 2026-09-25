import numpy as np

class VirtualKVCacheManager:
    def __init__(self, num_physical_blocks: int, block_size: int, hidden_dim: int):
        """
        Initialize the virtual memory KV cache manager.

        Args:
            num_physical_blocks: Total number of physical memory blocks
            block_size: Number of token slots per block
            hidden_dim: Dimension of each key/value vector
        """
        self.num_physical_blocks = num_physical_blocks
        self.block_size = block_size
        self.hidden_dim = hidden_dim

        # Physical memory: keys and values for each block and slot
        self.keys = np.zeros((num_physical_blocks, block_size, hidden_dim), dtype=float)
        self.values = np.zeros((num_physical_blocks, block_size, hidden_dim), dtype=float)

        # Free physical blocks (kept sorted ascending)
        self.free_blocks = list(range(num_physical_blocks))

        # Page tables: seq_id -> list of physical block indices (logical order)
        self.page_tables = {}

    def allocate_block(self, seq_id: int) -> int:
        """
        Allocate a new physical block for seq_id.
        Returns the logical block index assigned, or -1 if no blocks are available.
        """
        if seq_id not in self.page_tables:
            self.page_tables[seq_id] = []

        if not self.free_blocks:
            return -1

        # Allocate the lowest available physical block
        phys_block = self.free_blocks.pop(0)
        self.page_tables[seq_id].append(phys_block)

        # The new logical block index is the last index in the page table
        return len(self.page_tables[seq_id]) - 1

    def write(self, seq_id: int, position: int, key: np.ndarray, value: np.ndarray) -> bool:
        """
        Write a KV pair at the given token position. Auto-allocates blocks as needed.
        Returns True if successful, False if out of memory.
        """
        key = np.asarray(key, dtype=float)
        value = np.asarray(value, dtype=float)

        if seq_id not in self.page_tables:
            self.page_tables[seq_id] = []

        logical_block = position // self.block_size
        offset = position % self.block_size

        # Allocate blocks until the required logical block exists
        while len(self.page_tables[seq_id]) <= logical_block:
            if self.allocate_block(seq_id) == -1:
                return False

        phys_block = self.page_tables[seq_id][logical_block]
        self.keys[phys_block, offset, :] = key
        self.values[phys_block, offset, :] = value
        return True

    def read(self, seq_id: int, position: int):
        """
        Read the KV pair at the given token position.
        Returns (key, value) as numpy arrays, or (None, None) if not available.
        """
        if seq_id not in self.page_tables:
            return (None, None)

        logical_block = position // self.block_size
        offset = position % self.block_size

        if logical_block >= len(self.page_tables[seq_id]):
            return (None, None)

        phys_block = self.page_tables[seq_id][logical_block]
        # Return copies to prevent external mutation of internal storage
        return (
            self.keys[phys_block, offset, :].copy(),
            self.values[phys_block, offset, :].copy()
        )

    def free_sequence(self, seq_id: int) -> int:
        """
        Release all blocks allocated to the sequence back to the pool.
        Physical memory is zeroed. Returns the number of blocks freed.
        """
        if seq_id not in self.page_tables:
            return 0

        phys_blocks = self.page_tables[seq_id]
        count = len(phys_blocks)

        for phys in phys_blocks:
            # Zero out the physical block
            self.keys[phys, :, :] = 0
            self.values[phys, :, :] = 0
            # Return to free list
            self.free_blocks.append(phys)

        # Keep free list sorted so that allocation always picks lowest index
        self.free_blocks.sort()
        del self.page_tables[seq_id]
        return count

    def get_num_free_blocks(self) -> int:
        """Return the number of unallocated physical blocks."""
        return len(self.free_blocks)

    def get_page_table(self, seq_id: int) -> list:
        """
        Return a list of physical block indices in logical block order for the sequence,
        or an empty list if the sequence has no allocations.
        """
        return list(self.page_tables.get(seq_id, []))