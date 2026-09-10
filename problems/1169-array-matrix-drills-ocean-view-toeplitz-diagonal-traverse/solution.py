def array_drills(op, data):
    if op == "ocean":
        if not data:
            return []
        result = []
        max_right = -1  # every building > this initially (heights are positive)
        for i in range(len(data) - 1, -1, -1):
            if data[i] > max_right:
                result.append(i)
                max_right = data[i]
        result.reverse()  # increasing index order
        return result

    elif op == "toeplitz":
        if not data or not data[0]:
            return True
        m, n = len(data), len(data[0])
        for i in range(1, m):
            for j in range(1, n):
                if data[i][j] != data[i - 1][j - 1]:
                    return False
        return True

    elif op == "diagonal":
        if not data or not data[0]:
            return []
        m, n = len(data), len(data[0])
        result = []
        for d in range(m + n - 1):
            i_start = max(0, d - (n - 1))
            i_end = min(m - 1, d)
            diag = [data[i][d - i] for i in range(i_start, i_end + 1)]
            if d % 2 == 0:
                diag.reverse()  # even: up-right (bottom-left → top-right)
            result.extend(diag)
        return result

    else:
        raise ValueError(f"Unknown op: {op}")