def normalize_path(cwd, dest, home="/home/user", env=None):
    if env is None:
        env = {}
    
    # Expand leading ~
    if dest == '~':
        dest = home
    elif dest.startswith('~/'):
        dest = home + dest[1:]
    
    # Determine absolute vs relative
    is_absolute = dest.startswith('/')
    
    # Split and expand $NAME components (expansion may add more components)
    parts = []
    for part in dest.split('/'):
        if len(part) > 1 and part[0] == '$':
            name = part[1:]
            val = env.get(name, '')
            parts.extend(val.split('/'))
        else:
            parts.append(part)
    
    # Build starting stack
    if is_absolute:
        stack = []
    else:
        # Normalize cwd into a stack (handle '..' etc. just in case)
        stack = []
        for p in cwd.split('/'):
            if p == '' or p == '.':
                continue
            if p == '..':
                if stack:
                    stack.pop()
            else:
                stack.append(p)
    
    # Apply the destination components
    for p in parts:
        if p == '' or p == '.':
            continue
        if p == '..':
            if stack:
                stack.pop()
        else:
            stack.append(p)
    
    return '/' + '/'.join(stack)