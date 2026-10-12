import math
import bisect


def build_kdtree(points, depth=0):
    if not points:
        return None
    dim = len(points[0])
    axis = depth % dim
    points_sorted = sorted(points, key=lambda p: p[axis])
    mid = len(points_sorted) // 2
    point = points_sorted[mid]
    left_points = points_sorted[:mid]
    right_points = points_sorted[mid + 1:]
    return {
        "point": point,
        "axis": axis,
        "left": build_kdtree(left_points, depth + 1),
        "right": build_kdtree(right_points, depth + 1),
    }


def nearest_neighbor(tree, query):
    if tree is None:
        return None
    best_point = [None]
    best_dist_sq = [float('inf')]

    def search(node):
        if node is None:
            return
        point = node["point"]
        axis = node["axis"]
        dist_sq = sum((a - b) ** 2 for a, b in zip(point, query))
        if dist_sq < best_dist_sq[0]:
            best_dist_sq[0] = dist_sq
            best_point[0] = point

        if query[axis] < point[axis]:
            near, far = node["left"], node["right"]
        else:
            near, far = node["right"], node["left"]

        search(near)

        gap = query[axis] - point[axis]
        if gap * gap < best_dist_sq[0]:
            search(far)

    search(tree)
    if best_point[0] is None:
        return None
    return (best_point[0], round(math.sqrt(best_dist_sq[0]), 4))


def k_nearest(tree, query, k):
    if tree is None or k <= 0:
        return []

    best = []  # list of (dist_sq, point), sorted ascending by dist_sq

    def search(node):
        if node is None:
            return
        point = node["point"]
        axis = node["axis"]
        dist_sq = sum((a - b) ** 2 for a, b in zip(point, query))

        bisect.insort(best, (dist_sq, point))
        if len(best) > k:
            best.pop()

        if query[axis] < point[axis]:
            near, far = node["left"], node["right"]
        else:
            near, far = node["right"], node["left"]

        search(near)

        if len(best) < k or (query[axis] - point[axis]) ** 2 < best[-1][0]:
            search(far)

    search(tree)
    best.sort(key=lambda x: (x[0], x[1]))
    return [(p, round(math.sqrt(d), 4)) for d, p in best]