def find_duplicates(records):
    # records: list of hashable items (ints or strings)
    # return a list of values that appear more than once,
    # each listed once, ordered by the position of its second occurrence
    seen = set()       # values encountered so far
    reported = set()   # values already added to the result
    result = []
    for x in records:
        if x in seen:
            if x not in reported:
                result.append(x)
                reported.add(x)
        else:
            seen.add(x)
    return result