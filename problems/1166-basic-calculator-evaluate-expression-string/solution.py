def evaluate_expression(s):
    n = len(s)
    i = 0
    
    def skip_spaces():
        nonlocal i
        while i < n and s[i] == ' ':
            i += 1
    
    def parse_expr():
        nonlocal i
        val = parse_term()
        skip_spaces()
        while i < n and s[i] in '+-':
            op = s[i]
            i += 1
            rhs = parse_term()
            if op == '+':
                val += rhs
            else:
                val -= rhs
            skip_spaces()
        return val
    
    def parse_term():
        nonlocal i
        val = parse_factor()
        skip_spaces()
        while i < n and s[i] in '*/':
            op = s[i]
            i += 1
            rhs = parse_factor()
            if op == '*':
                val *= rhs
            else:
                # truncate toward zero
                val = int(val / rhs)  # careful with floats
            skip_spaces()
        return val
    
    def parse_factor():
        nonlocal i
        skip_spaces()
        if i < n and s[i] == '(':
            i += 1
            val = parse_expr()
            skip_spaces()
            if i < n and s[i] == ')':
                i += 1
            return val
        if i < n and s[i] == '-':
            i += 1
            return -parse_factor()
        # number
        start = i
        while i < n and s[i].isdigit():
            i += 1
        return int(s[start:i])
    
    return parse_expr()