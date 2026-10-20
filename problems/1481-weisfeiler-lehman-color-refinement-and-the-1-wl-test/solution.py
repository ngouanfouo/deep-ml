from collections import Counter


def wl_refine(adj, colors):
    signatures = []
    for i in range(len(adj)):
        neigh_colors = tuple(sorted(colors[j] for j in adj[i]))
        signatures.append((colors[i], neigh_colors))

    distinct = sorted(set(signatures))
    label = {sig: idx for idx, sig in enumerate(distinct)}
    return [label[sig] for sig in signatures]


def wl_iterate(adj, colors=None):
    if colors is None:
        colors = [0] * len(adj)
    else:
        colors = list(colors)

    rounds = 0
    while True:
        new_colors = wl_refine(adj, colors)
        if len(set(new_colors)) > len(set(colors)):
            colors = new_colors
            rounds += 1
        else:
            colors = new_colors
            break

    return colors, rounds


def color_histogram(colors):
    return sorted(Counter(colors).values())


def disjoint_union(adj1, adj2):
    n1 = len(adj1)
    union = [list(adj1[i]) for i in range(n1)]
    for j in range(len(adj2)):
        union.append([n + n1 for n in adj2[j]])
    return union


def wl_test(adj1, adj2):
    if len(adj1) != len(adj2):
        return False

    n = len(adj1)
    union = disjoint_union(adj1, adj2)
    colors, _ = wl_iterate(union)

    return sorted(colors[:n]) == sorted(colors[n:])