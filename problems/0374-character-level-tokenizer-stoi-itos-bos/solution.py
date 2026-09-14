import torch
import torch.nn as nn

class CharTokenizer:
    def __init__(self, text: str, embedding_dim: int = 32):
        """
        Build a character-level tokenizer from the input text.
        Includes a PyTorch nn.Embedding layer for the full vocabulary.

        Args:
            text: A string used to build the vocabulary.
            embedding_dim: Dimension of the embedding layer (default: 32).
        """
        # Extract unique characters and sort them in ascending ASCII order
        unique_chars = sorted(list(set(text)))
        
        # Initialize dictionaries for token-to-index and index-to-token mappings
        self.stoi = {'<BOS>': 0, '<EOS>': 1}
        self.itos = {0: '<BOS>', 1: '<EOS>'}
        
        # Populate mappings for unique characters starting from index 2
        for idx, char in enumerate(unique_chars, start=2):
            self.stoi[char] = idx
            self.itos[idx] = char
            
        # Set vocabulary size
        self.vocab_size = len(self.stoi)
        
        # Initialize the PyTorch embedding layer for the full vocabulary
        self.embedding = nn.Embedding(self.vocab_size, embedding_dim)

    def encode(self, text: str) -> torch.Tensor:
        """
        Encode a string into a 1D LongTensor of token indices,
        with BOS prepended and EOS appended.

        Args:
            text: The string to encode.
        Returns:
            torch.Tensor of dtype torch.long.
        """
        # Start with the BOS token index (0)
        indices = [0]
        
        # Map each character to its vocabulary index
        for char in text:
            if char in self.stoi:
                indices.append(self.stoi[char])
            else:
                raise ValueError(f"Character '{char}' not found in tokenizer vocabulary.")
                
        # Append the EOS token index (1)
        indices.append(1)
        
        return torch.tensor(indices, dtype=torch.long)

    def decode(self, indices) -> str:
        """
        Decode a 1D LongTensor (or list/iterable) of token indices back into a string.

        Args:
            indices: torch.Tensor or list of integer indices.
        Returns:
            Decoded string.
        """
        # Convert tensor to a standard Python list if necessary
        if isinstance(indices, torch.Tensor):
            indices = indices.tolist()
            
        # Look up each index in itos and concatenate the results
        decoded_tokens = [self.itos[int(idx)] for idx in indices]
        
        return "".join(decoded_tokens)