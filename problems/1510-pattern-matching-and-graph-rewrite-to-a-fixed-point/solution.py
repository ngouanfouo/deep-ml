class UPat:
    CONST = "const"
    VAR = "var"

    def __init__(self, op=None, src=None, name=None):
        self.op = op
        self.src = src
        self.name = name

    def match(self, expr, store):
        trial = dict(store)
        if self._match(expr, trial):
            store.update(trial)
            return True
        return False

    def _match(self, expr, store):
        if isinstance(expr, bool):
            return False
        if isinstance(expr, (int, float)):
            expr_kind = self.CONST
            children = ()
        elif isinstance(expr, str):
            expr_kind = self.VAR
            children = ()
        elif isinstance(expr, tuple) and len(expr) > 0:
            expr_kind = expr[0]
            children = expr[1:]
        else:
            return False

        if self.op is not None:
            if isinstance(self.op, set):
                if expr_kind not in self.op:
                    return False
            else:
                if expr_kind != self.op:
                    return False

        if self.src is not None:
            if len(self.src) != len(children):
                return False
            for pat, child in zip(self.src, children):
                if not pat._match(child, store):
                    return False

        if self.name is not None:
            if self.name in store:
                if store[self.name] != expr:
                    return False
            else:
                store[self.name] = expr

        return True


def rewrite(expr, rules):
    memo = {}
    count = [0]

    def rw(e):
        if e in memo:
            return memo[e]

        if isinstance(e, tuple) and len(e) > 0:
            new_children = tuple(rw(c) for c in e[1:])
            new_e = (e[0],) + new_children
        else:
            new_e = e

        while True:
            applied = False
            for pat, fn in rules:
                store = {}
                if pat.match(new_e, store):
                    result = fn(**store)
                    if result is not None and result != new_e:
                        count[0] += 1
                        new_e = rw(result)
                        applied = True
                        break
            if not applied:
                break

        memo[e] = new_e
        return new_e

    result = rw(expr)
    return result, count[0]