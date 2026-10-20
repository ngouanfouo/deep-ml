import numpy as np


def submission_score(outcomes, penalty_medians):
    """
    Score of a submission sequence (booleans, in order): penalty_medians[min(k, len - 1)] for the first
    solved submission after k failures, else 0.0. Report section 5.3.1.
    """
    failures = 0
    for solved in outcomes:
        if solved:
            return float(penalty_medians[min(failures, len(penalty_medians) - 1)])
        failures += 1
    return 0.0


def contest_rank(score, human_scores):
    """
    1 + number of human contest scores strictly greater than score.
    """
    return 1 + sum(1 for h in human_scores if h > score)


def performance_rating(rank, human_ratings, lo=0.0, hi=4000.0, iters=50):
    """
    Rating R with seed(R) = 1 + sum_j 1 / (1 + 10 ** ((R - r_j) / 400)) equal to rank,
    by `iters` bisection steps on [lo, hi]; returns the final midpoint.
    """
    human_ratings = np.asarray(human_ratings, dtype=float)

    def seed(R):
        return 1.0 + np.sum(1.0 / (1.0 + 10.0 ** ((R - human_ratings) / 400.0)))

    mid = 0.5 * (lo + hi)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if seed(mid) > rank:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def estimate_elo(results, penalty_medians, human_scores, human_ratings, n_samples=100, n_pick=10, seed=0):
    """
    Monte-Carlo expected rating: n_samples contests, each sampling n_pick of the candidates per problem
    without replacement in random order with np.random.default_rng(seed).
    Returns {"expected_rating", "mean_score", "min_rating", "max_rating"} rounded to 2 decimals.
    """
    rng = np.random.default_rng(seed)

    ratings = []
    scores = []

    for _ in range(n_samples):
        total = 0.0
        for p in range(len(results)):
            n = len(results[p])
            k = min(n_pick, n)
            order = rng.permutation(n)[:k]
            sampled = [results[p][i] for i in order]
            total += submission_score(sampled, penalty_medians[p])

        rank = contest_rank(total, human_scores)
        rating = performance_rating(rank, human_ratings)

        scores.append(total)
        ratings.append(rating)

    ratings = np.asarray(ratings, dtype=float)
    scores = np.asarray(scores, dtype=float)

    return {
        "expected_rating": round(float(ratings.mean()), 2),
        "mean_score": round(float(scores.mean()), 2),
        "min_rating": round(float(ratings.min()), 2),
        "max_rating": round(float(ratings.max()), 2),
    }