import torch

def cache_aware_route(replicas: list, requests: list, alpha: float = 0.7, beta: float = 0.3) -> dict:
    """
    Route inference requests to model replicas based on cache affinity and load using PyTorch.
    """
    # Work on local copies to avoid mutating input
    reps = []
    for r in replicas:
        reps.append({
            'cached_prefixes': r['cached_prefixes'],
            'active_requests': r['active_requests'],
            'max_capacity': r['max_capacity']
        })

    assignments = []
    hit_rates = []

    for req in requests:
        tokens = req['tokens']
        req_len = len(tokens)

        # Check if any replica has capacity left
        non_full_exists = any(r['active_requests'] < r['max_capacity'] for r in reps)

        best_score = -float('inf')
        best_idx = -1
        best_hit = 0.0

        for idx, r in enumerate(reps):
            # Skip full replicas if there is at least one non‑full one
            if non_full_exists and r['active_requests'] >= r['max_capacity']:
                continue

            # Find longest common prefix among all cached prefixes on this replica
            max_match = 0
            for prefix in r['cached_prefixes']:
                common = 0
                min_len = min(len(tokens), len(prefix))
                while common < min_len and tokens[common] == prefix[common]:
                    common += 1
                if common > max_match:
                    max_match = common

            hit_rate = max_match / req_len if req_len > 0 else 0.0
            load_ratio = r['active_requests'] / r['max_capacity'] if r['max_capacity'] > 0 else 0.0
            score = alpha * hit_rate - beta * load_ratio

            # Tie‑break: keep the first (smallest index) best score
            if score > best_score:
                best_score = score
                best_idx = idx
                best_hit = hit_rate

        # Assign request to the chosen replica
        assignments.append(best_idx)
        hit_rates.append(round(best_hit, 2))
        reps[best_idx]['active_requests'] += 1

    avg_hit = sum(hit_rates) / len(hit_rates) if hit_rates else 0.0
    load_dist = [r['active_requests'] for r in reps]

    return {
        'assignments': assignments,
        'cache_hit_rates': hit_rates,
        'avg_cache_hit_rate': round(avg_hit, 2),
        'load_distribution': load_dist
    }