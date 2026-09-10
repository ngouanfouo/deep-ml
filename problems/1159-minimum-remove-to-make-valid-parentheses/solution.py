def min_remove_to_make_valid(s):
    # s is a string of lowercase letters, '(' and ')'
    # Return a valid string after removing the fewest parentheses
    stack = []       # indices of unmatched '('
    remove = set()   # indices to delete
    
    # Pass 1: match parentheses, mark unmatched ones
    for i, ch in enumerate(s):
        if ch == '(':
            stack.append(i)
        elif ch == ')':
            if stack:
                stack.pop()      # matched with an earlier '('
            else:
                remove.add(i)    # unmatched ')'
    
    # Any '(' still on the stack never found a match
    remove.update(stack)
    
    # Pass 2: rebuild, skipping marked indices
    return ''.join(ch for i, ch in enumerate(s) if i not in remove)