class UOp:
    _cache = {}

    def __new__(cls, op, src=(), arg=None):
        src = tuple(src)
        key = (op, src, arg)
        if key in cls._cache:
            return cls._cache[key]

        obj = super().__new__(cls)
        obj.op = op
        obj.src = src
        obj.arg = arg
        cls._cache[key] = obj
        return obj

    def __init__(self, op, src=(), arg=None):
        # All initialization is done in __new__ so cached objects are not mutated.
        pass

    @staticmethod
    def const(v):
        return UOp("const", (), v)

    @staticmethod
    def var(name):
        return UOp("var", (), name)

    def _wrap(self, other):
        return other if isinstance(other, UOp) else UOp.const(other)

    def __add__(self, other):
        return UOp("add", (self, self._wrap(other)))

    def __radd__(self, other):
        return UOp("add", (self._wrap(other), self))

    def __mul__(self, other):
        return UOp("mul", (self, self._wrap(other)))

    def __rmul__(self, other):
        return UOp("mul", (self._wrap(other), self))

    def __floordiv__(self, other):
        return UOp("idiv", (self, self._wrap(other)))

    def __rfloordiv__(self, other):
        return UOp("idiv", (self._wrap(other), self))

    def toposort(self):
        out = []
        visited = set()
        stack = [(self, False)]

        while stack:
            node, expanded = stack.pop()

            if expanded:
                if node not in visited:
                    visited.add(node)
                    out.append(node)
            else:
                if node in visited:
                    continue

                stack.append((node, True))
                for child in reversed(node.src):
                    if child not in visited:
                        stack.append((child, False))

        return out


def node_count(u):
    seen = set()
    stack = [u]

    while stack:
        node = stack.pop()
        if node in seen:
            continue
        seen.add(node)
        stack.extend(node.src)

    return len(seen)


def tree_size(u, memo=None):
    if memo is None:
        memo = {}

    if u in memo:
        return memo[u]

    stack = [(u, False)]

    while stack:
        node, expanded = stack.pop()

        if node in memo:
            continue

        if expanded:
            total = 1
            for child in node.src:
                total += memo[child]
            memo[node] = total
        else:
            stack.append((node, True))
            for child in node.src:
                if child not in memo:
                    stack.append((child, False))

    return memo[u]