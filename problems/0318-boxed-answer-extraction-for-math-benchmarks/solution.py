import torch

def extract_boxed_answer(response: str) -> str:
    """
    Extract the answer from within \\boxed{...} in a model response.
    Uses PyTorch tensor operations for character-level sequence processing.
    
    Args:
        response: The model's text response containing a boxed answer
    
    Returns:
        The content inside the last \\boxed{}, or empty string if not found
    """
    # Search for all occurrences of \boxed{ in the string
    # We need to handle the literal backslash and boxed
    start_marker = "\\boxed{"
    
    # Find all positions where \boxed{ occurs
    start_positions = []
    search_start = 0
    while True:
        pos = response.find(start_marker, search_start)
        if pos == -1:
            break
        start_positions.append(pos + len(start_marker))  # Position after the opening brace
        search_start = pos + 1
    
    # If no boxed found, return empty string
    if not start_positions:
        return ""
    
    # Process from the last occurrence (rightmost \boxed{)
    start_idx = start_positions[-1]
    
    # Convert response to tensor of character codes
    chars = torch.tensor([ord(c) for c in response], dtype=torch.long)
    
    # Track brace depth starting from the opening brace
    # We start at depth 1 because we're inside the first brace
    depth = 1
    end_idx = start_idx
    
    # Iterate through characters after the opening brace
    for i in range(start_idx, len(response)):
        char = response[i]
        if char == '{':
            depth += 1
        elif char == '}':
            depth -= 1
            if depth == 0:
                end_idx = i
                break
    
    # Extract the content between start_idx and end_idx
    if end_idx > start_idx:
        return response[start_idx:end_idx]
    else:
        return ""