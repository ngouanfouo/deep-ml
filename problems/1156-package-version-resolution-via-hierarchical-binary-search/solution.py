def resolve_version(versions, max_major, max_minor):
    # versions: list of [major, minor]
    # return latest supported [major, minor], or None
    supported = [
        v for v in versions
        if v[0] < max_major or (v[0] == max_major and v[1] <= max_minor)
    ]
    if not supported:
        return None
    # Natural version ordering: compare major first, then minor
    best = max(supported, key=lambda v: (v[0], v[1]))
    return [best[0], best[1]]