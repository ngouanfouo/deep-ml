import numpy as np

# Transition table: (state, symbol) -> next state
REBER = {
    (0, 'B'): 1,
    (1, 'T'): 2,
    (1, 'P'): 3,
    (2, 'S'): 2,
    (2, 'X'): 4,
    (3, 'T'): 3,
    (3, 'V'): 5,
    (4, 'X'): 3,
    (4, 'S'): 6,
    (5, 'P'): 4,
    (5, 'V'): 6,
    (6, 'E'): 7,
}
FINAL = 7


def reber_next(state):
    """Sorted list of symbols allowed from the given state."""
    symbols = [sym for (s, sym) in REBER if s == state]
    return sorted(symbols)


def reber_string(rng):
    """Generate a random Reber string from state 0 to state 7."""
    state = 0
    chars = []
    while state != FINAL:
        options = reber_next(state)
        idx = int(rng.integers(0, len(options)))
        sym = options[idx]
        chars.append(sym)
        state = REBER[(state, sym)]
    return ''.join(chars)


def is_reber(s):
    """Return True if s is accepted by the Reber machine."""
    state = 0
    for c in s:
        if (state, c) not in REBER:
            return False
        state = REBER[(state, c)]
    return state == FINAL


def embedded_reber(rng):
    """Generate an embedded Reber string."""
    outer = 'T' if rng.integers(0, 2) == 0 else 'P'
    inner = reber_string(rng)
    return 'B' + outer + inner + outer + 'E'


def is_embedded_reber(s):
    """Check whether s is a valid embedded Reber string."""
    if len(s) < 5:
        return False
    if s[0] != 'B' or s[-1] != 'E':
        return False
    outer = s[1]
    if outer not in ('T', 'P'):
        return False
    if s[-2] != outer:
        return False
    inner = s[2:-2]
    return is_reber(inner)


def next_symbols(prefix):
    """
    Return the sorted list of symbols that can follow a valid, incomplete
    prefix of an embedded Reber string. Return [] if the prefix is invalid
    or already complete.
    """
    phase = 0          # 0: before B, 1: after B, 2: inside inner, 3: closing outer, 4: E, 5: done
    outer = None
    reber_state = 0

    for c in prefix:
        if phase == 0:
            if c == 'B':
                phase = 1
            else:
                return []
        elif phase == 1:
            if c in ('T', 'P'):
                outer = c
                phase = 2
                reber_state = 0
            else:
                return []
        elif phase == 2:
            if (reber_state, c) in REBER:
                reber_state = REBER[(reber_state, c)]
                if reber_state == FINAL:
                    phase = 3
            else:
                return []
        elif phase == 3:
            if c == outer:
                phase = 4
            else:
                return []
        elif phase == 4:
            if c == 'E':
                phase = 5
            else:
                return []
        else:  # phase == 5
            return []

    if phase == 0:
        return ['B']
    if phase == 1:
        return ['P', 'T']
    if phase == 2:
        return reber_next(reber_state)
    if phase == 3:
        return [outer]
    if phase == 4:
        return ['E']
    return []  # phase 5: complete